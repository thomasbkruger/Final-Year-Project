/*
  Wind-tunnel collimator wedge + beam-block cut patterns  (Phase D)
  ------------------------------------------------------------------
  Test section 500 (x, flow) x 250 (y) x 250 (z) mm inside.
  Thorlabs F810APC-1550 on the floor, beam 35 deg above the flow, pointing
  downstream; the beam ends in a matte-black plate over a window in the ceiling.

  PART = "wedge"        one-piece PLA wedge, already in print orientation (on its side)
         "floor_cut"    2-D: fibre slot + 4 bolt holes (tunnel coordinates, export DXF/SVG)
         "ceiling_cut"  2-D: 100 x 60 window + 4 bolt holes
         "block_plate"  2-D: 140 x 100 beam-block plate (laser-cut, paint the underside matte black)
         "assembly"     check only

  Tunnel coordinates for the cut patterns: x from the INLET end of the test
  section, y from the centre line, both on the inside faces of the panels.

  Wedge frame: origin on the floor surface directly below the lens centre,
  x downstream, z up.  Lens centre 26 mm above the floor.  The 35 deg top face
  is parallel to the beam: a phone inclinometer laid on it reads the beam angle.

  Print: on its side as exported, PLA, 0.2 mm layers, 4 perimeters, 25 % gyroid,
  no supports.  All horizontal holes are capped teardrops (bridges <= 8 mm).

  Hardware
    wedge  : 2x M3x35 (front) + 2x M3x16 (rear) socket head, from the top,
             4x M3 washer + nut under the floor   (lengths for a 3-6 mm floor)
             1x M3x8 grub screw (sits just below the top face) - clamps the collimator barrel;
             it self-threads into the 2.5 mm hole, so run it in once before fitting
    block  : 4x M4x20 button head from inside the tunnel, 4x M4 washer + wing nut on top
             matte black spray paint (carbon-black "flat/matt black") for the plate's underside

  This file mirrors model.py, which generated the STL and the DXF/SVG files and
  was used for all clearance and wall-thickness checks.
*/

PART = "assembly";

// ---------------- geometry ----------------
E      = 35;      // beam elevation [deg]
ZL     = 26;      // lens centre above the floor surface
H_TOP  = 15;      // top face: perpendicular offset above the beam axis
W      = 46;      // wedge width
NOSE_CUT = 1.5;   // blunt the nose feather edge

// F810APC-1550 (s = distance behind the lens face)
HEAD_D = 24.0;  HEAD_L = 15.5;
BARREL_D = 15.0; BARREL_S0 = 20.1; BARREL_S1 = 36.7;
COLL_L = 46.8;  REAR_D = 11.0;
HEAD_BORE = 24.4; BARREL_BORE = 15.3; CHANNEL_D = 16;

// fasteners in the wedge
BOLT_Y = 16.5;  BOLT_X = [-24, -48];
BOLT_HOLE = 3.4; CB_D = 6.6; CB_DEPTH = 5.6;
SS_S = 24;      SS_PILOT = 2.5;

// tunnel
BOX_L = 500; BOX_W = 250; BOX_H = 250;
X_LENS = 105;                         // lens face from the inlet end
X_WIN = 430; WIN_L = 100; WIN_W = 60;
PLATE_L = 140; PLATE_W = 100; PLATE_HOLE_DX = 60; PLATE_HOLE_DY = 40; PLATE_HOLE = 4.5;
// fibre slot (wedge frame) - covers a floor up to 6 mm thick, 1 mm margin
SLOT_X0 = -60.5; SLOT_X1 = -23.0; SLOT_HW = 8.8;

$fn = 96;

// ---------------- derived ----------------
T      = [-H_TOP * sin(E), ZL + H_TOP * cos(E)];    // top edge at the lens face [x, z]
X_NOSE = T[0] - T[1] / tan(E);
X_TAIL = ZL * tan(E);
function z_top(x)  = T[1] + (x - T[0]) * tan(E);
function z_axis(x) = ZL + x * tan(E);
SS_X = -SS_S * cos(E);

// local frame on the collimator axis: +z = beam direction, origin = lens centre
module coll_frame() translate([0, 0, ZL]) rotate([0, 90 - E, 0]) children();

// teardrop with its point toward +y (print-up), optionally flat-capped at y = cap
module teardrop2d(r, cap = -1) {
    c = (cap < 0) ? r * sqrt(2) : min(cap, r * sqrt(2));
    union() {
        circle(r);
        intersection() {
            polygon([[r / sqrt(2), r / sqrt(2)], [0, r * sqrt(2)], [-r / sqrt(2), r / sqrt(2)]]);
            translate([-r, -r]) square([2 * r, r + c]);
        }
    }
}
module prof(r, plain) { if (plain) circle(r); else teardrop2d(r, r + 1); }

module coll_cavity(plain = false) coll_frame() {
    translate([0, 0, -HEAD_L]) linear_extrude(HEAD_L + 5) prof(HEAD_BORE / 2, plain);
    translate([0, 0, -BARREL_S0]) linear_extrude(BARREL_S0 - HEAD_L, scale = HEAD_BORE / BARREL_BORE) prof(BARREL_BORE / 2, plain);
    translate([0, 0, -BARREL_S1 - 0.01]) linear_extrude(BARREL_S1 - BARREL_S0 + 0.02) prof(BARREL_BORE / 2, plain);
    translate([0, 0, -200]) linear_extrude(200 - BARREL_S1) prof(CHANNEL_D / 2, plain);
}

module wedge_body()
    rotate([90, 0, 0]) translate([0, 0, -W / 2]) linear_extrude(W)
        polygon([[X_NOSE + NOSE_CUT, 0], [X_NOSE + NOSE_CUT, z_top(X_NOSE + NOSE_CUT)],
                 [T[0], T[1]], [X_TAIL, 0]]);

module vhole(x, y, r, z0, h, cap) translate([x, y, z0]) linear_extrude(h) teardrop2d(r, cap);

module fasteners() {
    for (xb = BOLT_X, yb = [-BOLT_Y, BOLT_Y]) {
        vhole(xb, yb, BOLT_HOLE / 2, -1, 60, BOLT_HOLE / 2 + 0.5);
        vhole(xb, yb, CB_D / 2, z_top(xb) - CB_DEPTH, 40, CB_D / 2 + 0.5);
    }
    vhole(SS_X, 0, SS_PILOT / 2, z_axis(SS_X), 40, SS_PILOT / 2 + 0.5);   // clamp screw
}

module wedge() difference() { wedge_body(); coll_cavity(false); fasteners(); }

// ---------------- 2-D cut patterns (tunnel coordinates) ----------------
module stadium(x0, x1, hw) hull() { translate([x0 + hw, 0]) circle(hw); translate([x1 - hw, 0]) circle(hw); }
module rrect(l, w, r) offset(r) square([l - 2 * r, w - 2 * r], center = true);

module floor_cut() translate([X_LENS, 0]) {
    stadium(SLOT_X0, SLOT_X1, SLOT_HW);
    for (xb = BOLT_X, yb = [-BOLT_Y, BOLT_Y]) translate([xb, yb]) circle(d = BOLT_HOLE);
}
module plate_holes() for (sx = [-1, 1], sy = [-1, 1]) translate([sx * PLATE_HOLE_DX, sy * PLATE_HOLE_DY]) circle(d = PLATE_HOLE);
module ceiling_cut() translate([X_WIN, 0]) { rrect(WIN_L, WIN_W, 3); plate_holes(); }
module block_plate() difference() { rrect(PLATE_L, PLATE_W, 5); plate_holes(); }

// ---------------- assembly (check only) ----------------
module collimator_dummy() coll_frame() {
    color("LightSteelBlue") {
        translate([0, 0, -HEAD_L]) cylinder(d = HEAD_D, h = HEAD_L);
        translate([0, 0, -BARREL_S0]) cylinder(d1 = BARREL_D, d2 = HEAD_D, h = BARREL_S0 - HEAD_L);
        translate([0, 0, -BARREL_S1]) cylinder(d = BARREL_D, h = BARREL_S1 - BARREL_S0);
        translate([0, 0, -COLL_L]) cylinder(d = REAR_D, h = COLL_L - BARREL_S1);
    }
    color("DimGray") translate([0, 0, -COLL_L - 45]) cylinder(d = 11, h = 45);   // FC/APC connector
}

module assembly(t = 6) {
    color("Tan", 0.6) translate([0, -BOX_W / 2, -t]) linear_extrude(t)
        difference() { square([BOX_L, BOX_W]); translate([0, BOX_W / 2]) floor_cut(); }
    color("Tan", 0.6) translate([0, -BOX_W / 2, BOX_H]) linear_extrude(t)
        difference() { square([BOX_L, BOX_W]); translate([0, BOX_W / 2]) ceiling_cut(); }
    color([0.15, 0.15, 0.15]) translate([X_WIN, 0, BOX_H + t]) linear_extrude(6) block_plate();
    translate([X_LENS, 0, 0]) {
        color("Orange") wedge();
        collimator_dummy();
        color("Red", 0.6) coll_frame() cylinder(d = 7, h = (BOX_H - ZL) / sin(E) + 10);
    }
    %translate([0, -BOX_W / 2, 0]) cube([BOX_L, BOX_W, BOX_H]);
}

if      (PART == "wedge")       translate([0, 0, W / 2]) rotate([90, 0, 0]) wedge();
else if (PART == "floor_cut")   floor_cut();
else if (PART == "ceiling_cut") ceiling_cut();
else if (PART == "block_plate") block_plate();
else                            assembly();
