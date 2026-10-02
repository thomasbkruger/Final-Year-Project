/*
  Stepper speed profile: ramp up -> hold -> ramp down -> stop
  Arduino UNO R4 Minima (Renesas RA4M1, 48 MHz) + Pololu DRV8825 carrier

  Runs the profile once after power-up. Press the board's RESET button
  (single press) to run it again. (A quick double press puts the R4 into
  bootloader mode instead - just press once.)

  Default profile: 0 -> 300 RPM at 60 RPM/s (5 s), hold 10 s, 300 -> 0 RPM (5 s).

  Step pulses come from a 32-bit GPT hardware timer, so the step rate during
  the hold is jitter-free and independent of loop() timing.

  WIRING
    Arduino D2  -> DRV8825 STEP   (D2 = GPT1 channel A, 32-bit timer)
    Arduino D8  -> DRV8825 DIR
    Arduino GND -> DRV8825 GND (common with VMOT supply GND)
    DRV8825 SLEEP + RESET -> 3.3V
    DRV8825 M2 -> 3.3V for 1/16 microstepping (M0, M1 unconnected = low)
        full: L L L | 1/2: H L L | 1/4: L H L | 1/8: H H L | 1/16: L L H | 1/32: H L H
    DRV8825 VMOT/GND -> supply, 100 uF electrolytic across it at the board
    Motor coils -> A1/A2 and B1/B2

  !! MICROSTEPS below MUST match the M0/M1/M2 wiring. If the code says 16 but
  !! the driver is in full step, the motor is driven 16x too fast and stalls.

  Serial Monitor (115200 baud) prints each phase and the exact hold speed.
*/

#include "pwm.h"

// ---------------- Profile settings ----------------
const float         HOLD_RPM        = 600.0;   // target speed (negative = reverse)
const unsigned long HOLD_MS         = 30000;   // time at full speed
const float         ACCEL_RPM_PER_S = 100.0;    // ramp rate up and down
const unsigned long START_DELAY_MS  = 2000;    // pause after reset before moving

// ---------------- Pins ----------------
const uint8_t STEP_PIN = 2;   // GPT0/GPT1 pin (D2, D3, D6 or D7) for 32-bit timing
const uint8_t DIR_PIN  = 8;

// ---------------- Motor / driver ----------------
const uint16_t FULL_STEPS_PER_REV = 200;   // 1.8 deg/step NEMA 17
const uint8_t  MICROSTEPS         = 16;    // must match M0/M1/M2 wiring
const double   STEPS_PER_REV      = (double)FULL_STEPS_PER_REV * MICROSTEPS;

const double MIN_STEP_HZ   = 0.2;   // below this the motor is treated as stopped
const double STEP_PULSE_US = 4.0;   // STEP high time (DRV8825 min 1.9 us)

// ---------------- State ----------------
enum Phase { WAITING, RAMP_UP, HOLDING, RAMP_DOWN, DONE };
Phase phase = WAITING;

PwmOut   stepPwm(STEP_PIN);
bool     pwmBegun   = false;
uint32_t timerClkHz = 48000000UL;
uint32_t pulseCounts;
uint32_t minPeriodCounts;

float    targetRPM  = 0.0;
float    currentRPM = 0.0;
bool     running    = false;
uint32_t curCounts  = 0;
unsigned long lastRampMs = 0;
unsigned long phaseStartMs = 0;

// ---------------- Timer helpers ----------------

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
    ok = stepPwm.begin(counts, pulseCounts, true, TIMER_SOURCE_DIV_1);
    pwmBegun = ok;
  } else {
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
    delayMicroseconds(5);
    if (!startTimer(counts)) return;
  } else if (counts != curCounts) {
    stepPwm.period_raw((int)counts);   // buffered: applies at end of current step
  }
  curCounts = counts;
}

// Moves currentRPM toward targetRPM at ACCEL_RPM_PER_S, every 5 ms
void rampTick() {
  unsigned long now = millis();
  if (now - lastRampMs < 5) return;
  float dt = (now - lastRampMs) * 0.001f;
  lastRampMs = now;

  float maxDelta = ACCEL_RPM_PER_S * dt;
  float err = targetRPM - currentRPM;
  if (fabs(err) <= maxDelta) currentRPM = targetRPM;   // land exactly on target
  else currentRPM += (err > 0 ? maxDelta : -maxDelta);

  applyRPM(currentRPM);
}

// ---------------- Reporting ----------------

void printSeconds() {
  Serial.print(F("["));
  Serial.print(millis() / 1000.0, 2);
  Serial.print(F(" s] "));
}

void printHoldSpeed() {
  double hz = actualStepHz(curCounts);
  double actual = hz * 60.0 / STEPS_PER_REV;
  if (currentRPM < 0) actual = -actual;
  Serial.print(F("  actual speed "));
  Serial.print(actual, 5);
  Serial.print(F(" RPM ("));
  Serial.print(hz, 4);
  Serial.print(F(" steps/s, period "));
  Serial.print(curCounts);
  Serial.println(F(" counts)"));
}

// ---------------- Main ----------------

void setup() {
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, LOW);

  timerClkHz      = R_FSP_SystemClockHzGet(FSP_PRIV_CLOCK_PCLKD);
  pulseCounts     = (uint32_t)(timerClkHz * STEP_PULSE_US * 1e-6 + 0.5);
  minPeriodCounts = 2 * pulseCounts;

  Serial.begin(115200);

  Serial.println(F("DRV8825 speed profile (UNO R4 Minima)"));
  Serial.print(F("Profile: 0 -> "));
  Serial.print(HOLD_RPM, 1);
  Serial.print(F(" RPM at "));
  Serial.print(ACCEL_RPM_PER_S, 1);
  Serial.print(F(" RPM/s, hold "));
  Serial.print(HOLD_MS / 1000.0, 1);
  Serial.print(F(" s, ramp down | microsteps "));
  Serial.println(MICROSTEPS);

  phaseStartMs = millis();
  lastRampMs   = millis();
}

void loop() {
  unsigned long now = millis();

  switch (phase) {
    case WAITING:
      if (now - phaseStartMs >= START_DELAY_MS) {
        targetRPM = HOLD_RPM;
        lastRampMs = now;
        phase = RAMP_UP;
        printSeconds(); Serial.println(F("Ramping up"));
      }
      break;

    case RAMP_UP:
      rampTick();
      if (currentRPM == targetRPM) {
        phase = HOLDING;
        phaseStartMs = now;
        printSeconds(); Serial.println(F("Holding"));
        printHoldSpeed();
      }
      break;

    case HOLDING:
      if (now - phaseStartMs >= HOLD_MS) {
        targetRPM = 0;
        lastRampMs = now;
        phase = RAMP_DOWN;
        printSeconds(); Serial.println(F("Ramping down"));
      }
      break;

    case RAMP_DOWN:
      rampTick();
      if (currentRPM == 0) {
        stopTimer();
        phase = DONE;
        printSeconds(); Serial.println(F("Stopped. Press RESET to run again."));
      }
      break;

    case DONE:
      break;   // motor stays energised (holding torque) until reset
  }
}
