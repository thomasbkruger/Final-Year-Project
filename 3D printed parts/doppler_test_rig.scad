/*
  Hard-target Doppler test rig
  - NEMA 17 stepper with 50 mm target disk on its shaft
  - Thorlabs F810APC-1550 fibre collimator in a split clamp
  - Beam horizontal, at 45 deg to the motor axis, 100 mm from the target,
    passing OFFSET_D above the motor axis.

  Line-of-sight velocity of the target:  v = omega * OFFSET_D * sin(BEAM_ANGLE)
  Doppler shift (monostatic):            f = 2 v / lambda
  With OFFSET_D = 15 mm, 45 deg, 1550 nm: f = 1.433 kHz per RPM (430 kHz at 300 RPM)

  Neither the disk's position along the shaft nor the collimator's position
  in its clamp changes OFFSET_D - it is set only by the printed heights of
  the motor axis and the clamp bore.

  Parts (set PART, or use -D PART="..." on the command line):
    "base"     - base plate + motor plate + collimator clamp (one piece)
    "disk"     - target disk with hub for the 5 mm D-shaft
    "assembly" - everything in place with dummy motor, collimator and beam
                 (for checking only, not for printing)

  Coordinates: bench = z 0. Motor axis along +X at height AXIS_H.
  Disk front face at x = 0. Beam hits the disk face at (0, 0, AXIS_H + OFFSET_D).

  Hardware:
    4x M3x8 socket screws        motor -> motor plate
    1x M3x20 socket screw + nut  collimator clamp
    1x M3x6 set screw + nut      disk hub (an M3x8 cap screw also works)
*/

PART = "assembly";   // "base", "disk" or "assembly"

// ---------------- Optical geometry ----------------
BEAM_ANGLE  = 45;    // angle between beam and motor axis [deg]
TARGET_DIST = 100;   // collimator front face to target spot, along beam [mm]
OFFSET_D    = 15;    // beam height above motor axis [mm] -> sets velocity

// ---------------- Base / motor ----------------
BASE_T  = 6;         // base plate thickness
AXIS_H  = 36;        // motor axis height above bench
NEMA_W    = 42.3;    // NEMA 17 body width
NEMA_HOLE = 31;      // mounting hole spacing
NEMA_BOSS = 22;      // centring boss diameter
M3_CLEAR  = 3.4;
PLATE_T   = 5;       // motor plate thickness
PLATE_W   = 56;      // motor plate width (y)
GUSSET_T  = 4;
GUSSET_L  = 35;
GUSSET_H  = 40;      // gusset height at the plate (above bench)

// ---------------- Disk ----------------
DISK_D  = 50;
DISK_T  = 3;
HUB_D   = 18;
HUB_L   = 10;
HUB_GAP = 2;         // gap between hub and motor plate
SHAFT_D = 5;
BORE_D  = 5.2;       // shaft bore (printed holes come out slightly small)
FLAT_R  = 2.1;       // centre-to-flat distance of D bore (shaft flat is ~2.0)

// ---------------- Collimator (Thorlabs F810APC-1550) ----------------
COLL_HEAD_D   = 24.0;
COLL_HEAD_L   = 15.5;
COLL_BARREL_D = 15.0;
BARREL_START  = 20.1;   // Ø15 barrel, measured from the front face
BARREL_END    = 36.7;
COLL_LEN      = 46.8;

// ---------------- Collimator clamp ----------------
CLAMP_BORE = 15.25;
CLAMP_LEN  = 12;        // must be shorter than the Ø15 barrel (16.6 mm)
CLAMP_W    = 26;
SLIT       = 1.5;
FOOT_LEN   = 44;        // clamp pillar footprint along the beam
FOOT_W     = 32;
FLARE_TOP  = 36;        // pillar flares out below this height (head bottom is at 39)
M3_HEAD_D  = 6.5;
M3_NUT_AF  = 5.8;       // M3 nut 5.5 across flats + clearance
POCKET     = 5;         // depth of screw-head counterbore and nut pocket

$fn = 96;

// ---------------- Derived ----------------
BEAM_Z      = AXIS_H + OFFSET_D;
PLATE_X1    = -(DISK_T + HUB_L + HUB_GAP);   // motor plate front face
PLATE_X0    = PLATE_X1 - PLATE_T;            // motor plate back face (motor side)
CLAMP_S     = TARGET_DIST + (BARREL_START + BARREL_END) / 2;  // clamp centre along beam
BOLT_Z      = BEAM_Z + CLAMP_BORE / 2 + 5;
CLAMP_TOP   = BOLT_Z + M3_HEAD_D / 2 + 2.5;
PLATE_TOP   = AXIS_H + NEMA_W / 2 + 5;

// Frame of the collimator clamp: origin under the clamp centre on the bench,
// local +x pointing along the beam away from the target (towards the fibre).
module at_clamp() {
  translate([CLAMP_S * cos(BEAM_ANGLE), CLAMP_S * sin(BEAM_ANGLE), 0])
    rotate([0, 0, BEAM_ANGLE]) children();
}

// Horizontal hole along +x with a 45 deg "roof" so it prints without support
module teardrop_x(d, len) {
  r = d / 2;
  hull() {
    rotate([0, 90, 0]) cylinder(d = d, h = len);
    translate([0, -0.01, r * sqrt(2) - 0.01]) cube([len, 0.02, 0.02]);
  }
}

// ================= BASE (one printed piece) =================

module base_plate() {
  hull() {
    translate([PLATE_X0 - GUSSET_L - 5, -PLATE_W / 2 - 2, 0])
      cube([GUSSET_L + 5 + PLATE_T + 3, PLATE_W + 4, BASE_T]);
    at_clamp() translate([-FOOT_LEN / 2, -FOOT_W / 2, 0])
      cube([FOOT_LEN, FOOT_W, BASE_T]);
  }
}

module motor_plate() {
  difference() {
    translate([PLATE_X0, -PLATE_W / 2, 0]) cube([PLATE_T, PLATE_W, PLATE_TOP]);
    // centring boss
    translate([PLATE_X0 - 1, 0, AXIS_H]) teardrop_x(NEMA_BOSS + 1, PLATE_T + 2);
    // 4 mounting holes
    for (sy = [-1, 1], sz = [-1, 1])
      translate([PLATE_X0 - 1, sy * NEMA_HOLE / 2, AXIS_H + sz * NEMA_HOLE / 2])
        rotate([0, 90, 0]) cylinder(d = M3_CLEAR, h = PLATE_T + 2, $fn = 32);
  }
  // gussets either side of the motor body
  for (sy = [-1, 1])
    translate([0, sy * (PLATE_W / 2 - GUSSET_T / 2), 0])
      hull() {
        translate([PLATE_X0 - 0.5, -GUSSET_T / 2, 0]) cube([0.5, GUSSET_T, GUSSET_H]);
        translate([PLATE_X0 - GUSSET_L, -GUSSET_T / 2, 0]) cube([0.5, GUSSET_T, BASE_T]);
      }
}

module collimator_clamp() {
  at_clamp() difference() {
    union() {
      // upright: only CLAMP_LEN long so it fits the Ø15 barrel behind the head
      translate([-CLAMP_LEN / 2, -CLAMP_W / 2, 0]) cube([CLAMP_LEN, CLAMP_W, CLAMP_TOP]);
      // flared foot for stiffness, kept below the collimator head
      hull() {
        translate([-CLAMP_LEN / 2, -CLAMP_W / 2, 0]) cube([CLAMP_LEN, CLAMP_W, FLARE_TOP]);
        translate([-FOOT_LEN / 2, -FOOT_W / 2, 0]) cube([FOOT_LEN, FOOT_W, BASE_T + 1]);
      }
    }
    // collimator bore (teardrop roof sits under the slit)
    translate([-FOOT_LEN, 0, BEAM_Z]) teardrop_x(CLAMP_BORE, 2 * FOOT_LEN);
    // clamping slit
    translate([-FOOT_LEN, -SLIT / 2, BEAM_Z]) cube([2 * FOOT_LEN, SLIT, CLAMP_TOP]);
    // M3 clamp screw, counterbore and nut pocket
    translate([0, 0, BOLT_Z]) rotate([90, 0, 0]) {
      cylinder(d = M3_CLEAR, h = CLAMP_W + 2, center = true, $fn = 32);
      translate([0, 0, CLAMP_W / 2 - POCKET]) cylinder(d = M3_HEAD_D, h = POCKET + 1, $fn = 48);
      translate([0, 0, -CLAMP_W / 2 - 1])
        cylinder(d = M3_NUT_AF / cos(30), h = POCKET + 1, $fn = 6);
    }
  }
}

module base() {
  base_plate();
  motor_plate();
  collimator_clamp();
}

// ================= TARGET DISK =================
// Modelled with the front (target) face at z = 0 - print it this way up.

module d_bore(h) {
  intersection() {
    cylinder(d = BORE_D, h = h, $fn = 48);
    translate([-BORE_D, -BORE_D, 0]) cube([2 * BORE_D, BORE_D + FLAT_R, h]);
  }
}

module disk() {
  screw_z = DISK_T + HUB_L / 2;
  nut_r   = 5.3;                              // radial position of the nut slot
  difference() {
    union() {
      cylinder(d = DISK_D, h = DISK_T, $fn = 180);
      cylinder(d = HUB_D, h = DISK_T + HUB_L);
    }
    translate([0, 0, -1]) d_bore(DISK_T + HUB_L + 2);
    // radial set screw, pointing at the shaft flat (+y)
    translate([0, 0, screw_z]) rotate([-90, 0, 0]) cylinder(d = 3.3, h = HUB_D, $fn = 32);
    // nut slot, open at the hub end (motor side)
    translate([-M3_NUT_AF / 2, nut_r - 1.35, screw_z - M3_NUT_AF / 2 - 0.3])
      cube([M3_NUT_AF, 2.7, HUB_L]);
  }
}

// ================= ASSEMBLY PREVIEW (not for printing) =================

module dummy_motor() {
  color("DimGray")
    translate([PLATE_X0 - 40, -NEMA_W / 2, AXIS_H - NEMA_W / 2]) cube([40, NEMA_W, NEMA_W]);
  color("Silver") {
    translate([PLATE_X0, 0, AXIS_H]) rotate([0, 90, 0]) cylinder(d = NEMA_BOSS, h = 2);
    translate([PLATE_X0, 0, AXIS_H]) rotate([0, 90, 0]) cylinder(d = SHAFT_D, h = 22);
  }
}

module dummy_collimator() {
  color("LightSteelBlue")
    translate([TARGET_DIST * cos(BEAM_ANGLE), TARGET_DIST * sin(BEAM_ANGLE), BEAM_Z])
      rotate([0, 0, BEAM_ANGLE]) rotate([0, 90, 0]) {
        cylinder(d = COLL_HEAD_D, h = COLL_HEAD_L);
        translate([0, 0, COLL_HEAD_L]) cylinder(d1 = COLL_HEAD_D, d2 = COLL_BARREL_D, h = BARREL_START - COLL_HEAD_L);
        translate([0, 0, BARREL_START]) cylinder(d = COLL_BARREL_D, h = BARREL_END - BARREL_START);
        translate([0, 0, BARREL_END]) cylinder(d = 11, h = COLL_LEN - BARREL_END);
      }
}

module beam() {
  color("Red", 0.6)
    translate([0, 0, BEAM_Z]) rotate([0, 0, BEAM_ANGLE]) rotate([0, 90, 0])
      cylinder(d = 7, h = TARGET_DIST);   // 7 mm 1/e^2 beam
}

module assembly() {
  color("Orange") base();
  color("White") translate([0, 0, AXIS_H]) rotate([0, -90, 0]) disk();
  dummy_motor();
  dummy_collimator();
  beam();
}

if (PART == "base") base();
else if (PART == "disk") disk();
else assembly();
