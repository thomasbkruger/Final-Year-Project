/*
  Constant-RPM stepper control
  Arduino UNO R4 Minima (Renesas RA4M1, 48 MHz) + Pololu DRV8825 carrier

  Step pulses come from a 32-bit GPT hardware timer in PWM mode, so the
  step rate is jitter-free and independent of loop() timing. The timer runs
  straight off the 48 MHz peripheral clock (no prescaler), which gives
  sub-ppm period resolution across the whole speed range. Speed changes are
  written to the timer's buffered period register and take effect cleanly at
  the end of the current step period (no glitches or dropped steps).

  WIRING
    Arduino D2  -> DRV8825 STEP   (D2 = GPT1 channel A, a 32-bit timer.
                                   D3 shares GPT1, so don't use D3 for PWM.)
    Arduino D8  -> DRV8825 DIR
    Arduino GND -> DRV8825 GND (logic GND, common with VMOT supply GND)
    DRV8825 SLEEP + RESET -> 3.3V (or 5V)
    DRV8825 M0/M1/M2 -> set microstepping to match MICROSTEPS below
        (pins are pulled low on the board; unconnected = full step)
        full: L L L | 1/2: H L L | 1/4: L H L | 1/8: H H L | 1/16: L L H | 1/32: H L H
    DRV8825 VMOT/GND -> 8.2-45 V supply, 100 uF electrolytic across it at the board
    Motor coils -> A1/A2 and B1/B2

    The UNO R4 Minima has 5 V logic outputs; the DRV8825 accepts 5 V logic.

  SERIAL COMMANDS (Serial Monitor, 115200 baud, newline or CR line ending)
    120      set target speed to 120 RPM (negative = reverse, e.g. -45.5)
    s        ramp down to stop
    x        immediate stop (no ramp)
    a60      set acceleration to 60 RPM/s (a0 = no ramp, instant change)
    ?        print status, including the actual RPM the timer produces
*/

#include "pwm.h"

// ---------------- Pins ----------------
const uint8_t STEP_PIN = 2;   // must be a GPT0/GPT1 pin (D2, D3, D6 or D7) for 32-bit timing
const uint8_t DIR_PIN  = 8;

// ---------------- Motor / driver ----------------
const uint16_t FULL_STEPS_PER_REV = 200;   // 1.8 deg/step NEMA 17
const uint8_t  MICROSTEPS         = 16;    // must match M0/M1/M2 jumpers
const double   STEPS_PER_REV      = (double)FULL_STEPS_PER_REV * MICROSTEPS;

// ---------------- Motion settings ----------------
const float  START_RPM     = 0.0;     // speed on power-up
const float  MAX_RPM       = 600.0;   // safety limit on commanded speed
const double MIN_STEP_HZ   = 0.2;     // below this the motor is treated as stopped
const double STEP_PULSE_US = 4.0;     // STEP high time (DRV8825 min 1.9 us)
float accelRPMperSec       = 120.0;   // ramp rate (0 = instant)

// ---------------- State ----------------
PwmOut stepPwm(STEP_PIN);
bool     pwmBegun   = false;
uint32_t timerClkHz = 48000000UL;     // replaced with the real value in setup()
uint32_t pulseCounts;                  // STEP high time in timer counts
uint32_t minPeriodCounts;              // fastest allowed step period

float    targetRPM  = START_RPM;
float    currentRPM = 0.0;
bool     running    = false;
bool     reportWhenReached = false;
uint32_t curCounts  = 0;
unsigned long lastRampMs = 0;

// ---------------- Timer helpers ----------------

// stepHz = timerClkHz / periodCounts
uint32_t countsFor(double stepHz) {
  double c = (double)timerClkHz / stepHz;
  if (c > 2147483647.0) c = 2147483647.0;
  uint32_t counts = (uint32_t)(c + 0.5);
  if (counts < minPeriodCounts) counts = minPeriodCounts;
  return counts;
}

double actualStepHz(uint32_t counts) {
  return (double)timerClkHz / (double)counts;
}

bool startTimer(uint32_t counts) {
  bool ok;
  if (!pwmBegun) {
    // First start: configure pin + timer (raw counts, no prescaler); this also starts it
    ok = stepPwm.begin(counts, pulseCounts, true, TIMER_SOURCE_DIV_1);
    pwmBegun = ok;
  } else {
    // Timer is stopped: period write applies immediately and resets the counter
    ok = stepPwm.period_raw((int)counts);
    stepPwm.resume();
  }
  running = ok;
  if (!ok) Serial.println(F("ERROR: could not start step timer (check STEP_PIN)"));
  return ok;
}

void stopTimer() {
  if (pwmBegun) stepPwm.suspend();
  running = false;
}

// While running, the period register is double-buffered by the GPT and
// transfers at the next counter overflow, so updates are glitch-free.
void updateTimer(uint32_t counts) {
  stepPwm.period_raw((int)counts);
}

// ---------------- Speed control ----------------

void applyRPM(float rpm) {
  double stepHz = fabs((double)rpm) * STEPS_PER_REV / 60.0;

  if (stepHz < MIN_STEP_HZ) {
    if (running) stopTimer();
    return;
  }

  uint32_t counts = countsFor(stepHz);

  if (!running) {
    digitalWrite(DIR_PIN, rpm > 0 ? HIGH : LOW);
    delayMicroseconds(5);        // DIR setup time before first STEP edge (min 650 ns)
    if (!startTimer(counts)) return;
  } else if (counts != curCounts) {
    updateTimer(counts);
  }
  curCounts = counts;
}

void rampTick() {
  unsigned long now = millis();
  if (now - lastRampMs < 5) return;
  float dt = (now - lastRampMs) * 0.001f;
  lastRampMs = now;

  float next;
  if (accelRPMperSec <= 0) {
    next = targetRPM;
  } else {
    float maxDelta = accelRPMperSec * dt;
    float err = targetRPM - currentRPM;
    if (err >  maxDelta) err =  maxDelta;
    if (err < -maxDelta) err = -maxDelta;
    next = currentRPM + err;
  }

  // Reversals always pass through a full stop so DIR only changes when idle
  if ((currentRPM > 0 && next < 0) || (currentRPM < 0 && next > 0)) next = 0;

  currentRPM = next;
  applyRPM(currentRPM);

  if (reportWhenReached && currentRPM == targetRPM) {
    reportWhenReached = false;
    printStatus();
  }
}

// ---------------- Serial interface ----------------

void printStatus() {
  Serial.print(F("Target "));
  Serial.print(targetRPM, 3);
  Serial.print(F(" RPM | current "));
  Serial.print(currentRPM, 3);
  if (running) {
    double hz = actualStepHz(curCounts);
    double actual = hz * 60.0 / STEPS_PER_REV;
    if (currentRPM < 0) actual = -actual;
    Serial.print(F(" | actual "));
    Serial.print(actual, 5);
    Serial.print(F(" RPM ("));
    Serial.print(hz, 4);
    Serial.print(F(" steps/s, period "));
    Serial.print(curCounts);
    Serial.print(F(" counts)"));
  } else {
    Serial.print(F(" | stopped"));
  }
  Serial.print(F(" | accel "));
  Serial.print(accelRPMperSec, 1);
  Serial.println(F(" RPM/s"));
}

void printHelp() {
  Serial.println(F("Commands: <rpm> (e.g. 120 or -45.5), s = ramp stop, x = instant stop,"));
  Serial.println(F("          a<rpm/s> = set accel (a0 = none), ? = status"));
}

void handleCommand(char *cmd) {
  while (*cmd == ' ') cmd++;

  switch (cmd[0]) {
    case 's': case 'S':
      targetRPM = 0;
      reportWhenReached = true;
      Serial.println(F("Ramping to stop"));
      return;
    case 'x': case 'X':
      targetRPM = 0;
      currentRPM = 0;
      stopTimer();
      Serial.println(F("Stopped"));
      return;
    case 'a': case 'A':
      accelRPMperSec = fabs(atof(cmd + 1));
      Serial.print(F("Accel set to "));
      Serial.print(accelRPMperSec, 1);
      Serial.println(F(" RPM/s"));
      return;
    case '?':
      printStatus();
      return;
  }

  char *end;
  double v = strtod(cmd, &end);
  if (end == cmd) {
    Serial.println(F("Unknown command"));
    printHelp();
    return;
  }
  if (fabs(v) > MAX_RPM) {
    Serial.print(F("Rejected: limit is +/-"));
    Serial.print(MAX_RPM, 0);
    Serial.println(F(" RPM (change MAX_RPM)"));
    return;
  }
  targetRPM = (float)v;
  reportWhenReached = true;
  Serial.print(F("Target set to "));
  Serial.print(targetRPM, 3);
  Serial.println(F(" RPM"));
}

char lineBuf[24];
uint8_t lineLen = 0;

void pollSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen) {
        lineBuf[lineLen] = '\0';
        handleCommand(lineBuf);
        lineLen = 0;
      }
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    }
  }
}

// ---------------- Main ----------------

void setup() {
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, LOW);

  // GPT counts from PCLKD (48 MHz on the UNO R4)
  timerClkHz      = R_FSP_SystemClockHzGet(FSP_PRIV_CLOCK_PCLKD);
  pulseCounts     = (uint32_t)(timerClkHz * STEP_PULSE_US * 1e-6 + 0.5);
  minPeriodCounts = 2 * pulseCounts;   // keeps STEP low time >= high time

  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 3000) { }   // wait for USB serial (max 3 s)

  Serial.println(F("DRV8825 constant-RPM controller (UNO R4 Minima)"));
  Serial.print(F("Timer clock: "));
  Serial.print(timerClkHz);
  Serial.print(F(" Hz | steps/rev: "));
  Serial.print((uint32_t)STEPS_PER_REV);
  Serial.print(F(" | max step rate: "));
  Serial.print(actualStepHz(minPeriodCounts), 0);
  Serial.println(F(" Hz"));
  printHelp();

  lastRampMs = millis();
  if (START_RPM != 0) reportWhenReached = true;
}

void loop() {
  pollSerial();
  rampTick();
}
