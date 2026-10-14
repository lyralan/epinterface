-- Optional effective-leakage-area infiltration and natural-ventilation control limits.
-- All columns are nullable; NULL keeps the previous behavior.
ALTER TABLE "Infiltration" ADD COLUMN "SpecificLeakageArea" REAL;
ALTER TABLE "Infiltration" ADD COLUMN "StackCoefficient" REAL;
ALTER TABLE "Infiltration" ADD COLUMN "WindCoefficient" REAL;
ALTER TABLE "Ventilation" ADD COLUMN "NaturalVentMinOutdoorTemperature" REAL;
ALTER TABLE "Ventilation" ADD COLUMN "NaturalVentMaxOutdoorTemperature" REAL;
ALTER TABLE "Ventilation" ADD COLUMN "NaturalVentDeltaTemperature" REAL;
ALTER TABLE "Ventilation" ADD COLUMN "NaturalVentAboveSetpointMidpoint" BOOLEAN;
