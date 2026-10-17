-- Optional AIM-2 flow-coefficient infiltration (ZoneInfiltration:FlowCoefficient).
-- All columns are nullable; NULL keeps the previous behavior.
ALTER TABLE "Infiltration" ADD COLUMN "FlowCoefficientPerFloorArea" REAL;
ALTER TABLE "Infiltration" ADD COLUMN "PressureExponent" REAL;
ALTER TABLE "Infiltration" ADD COLUMN "ShelterFactor" REAL;
