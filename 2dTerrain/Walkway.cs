namespace TerrainGenerator
{
    public class Walkway
    {
        public PointF start;
        public PointF end;
        private float width;
        private float height;
        private float angle;
        private int brickwidth;
        private int walkwaywidth;
        private int segments;

        public Walkway(PointF start, PointF end, float bend, int brickwidth, int walkwaywidth, int segments)
        {
            this.brickwidth = brickwidth;
            this.start = start;
            this.end = end;
            angle = end.Angle(start);

            height = (float)start.DistanceTo(end);
            width = bend * 2;
            this.walkwaywidth = walkwaywidth;
            this.segments = segments;
        }

        public void Draw(Bitmap result)
        {

            // Calculate the centre of the oval (for simplicity, we use the midpoint between the start and end points)
            PointF centre = new PointF((start.X + end.X) / 2, (start.Y + end.Y) / 2);

            // Calculate the semi-major and semi-minor axes based on the width and height
            float semi_major_axis = height / 2 - brickwidth;
            float semi_minor_axis = width / 2 - brickwidth;

            // Calculate the angle of rotation for the oval (assuming you want to rotate the oval based on the walkway's angle)
            float rotationOffset = angle;  // Or use another value for rotation

            // Calculate the angle increment for each segment
            DrawSide(result, centre, semi_major_axis, semi_minor_axis, rotationOffset, segments);

            var adjustedcentre = centre.Subtract(end.Subtract(start).Perpendicular().UnitVector().Scale(walkwaywidth / 4));
            semi_major_axis += walkwaywidth * 2;
            semi_minor_axis += walkwaywidth * 2;
            DrawSide(result, adjustedcentre, semi_major_axis, semi_minor_axis, rotationOffset, segments);

        }

        private void DrawSide(Bitmap result, PointF centre, float semi_major_axis, float semi_minor_axis, float rotationOffset, int segments)
        {
            var ovalestimation = GetOvalPoints(semi_major_axis, semi_minor_axis);
            double halflen = 0;
            for (int i = 0; i < ovalestimation.Count(); ++i)
            {
                var a = ovalestimation[i];
                var b = ovalestimation[i != ovalestimation.Count() - 1 ? i + 1 : 0];
                halflen += a.DistanceTo(b);
            }
            halflen /= 2;
            double expecteddistance = halflen / segments;
            // Draw the curved bricks along the rotated oval
            int lastdegreesidx = 0;
            int nextdegreesidx = 0;
            Graphics g = Graphics.FromImage(result);
            int segmentsdone = 0;
            do
            {
                double accumdst = 0;
                for (int i = lastdegreesidx; i < 180; ++i)
                {
                    accumdst += ovalestimation[i].DistanceTo(ovalestimation[i != ovalestimation.Count() - 1 ? i + 1 : 0]); //Find distance between me and the next point
                    nextdegreesidx = i;

                    if (accumdst > expecteddistance) //Found the correct arclen
                    {
                        break;
                    }
                }

                // Calculate the start and end positions based on the angle
                PointF start = ovalestimation[lastdegreesidx].Add(centre);
                PointF end = ovalestimation[nextdegreesidx].Add(centre);
                // Create and draw the curved brick using the CurvedBrick constructor
                CurvedBrick curvedBrick = new(
                    (int)semi_major_axis,
                    (int)semi_minor_axis,
                    start,
                    end,
                    centre,
                    brickwidth * 2,
                    rotationOffset,
                    (float)accumdst //Float required for drawing
                );
                curvedBrick.Draw(result);  // Draw the curved brick on the result bitmap
                
                lastdegreesidx = nextdegreesidx;
                segmentsdone++;
                //g.DrawEllipse(new Pen(Color.Red), new RectangleF(start.X - 5, start.Y - 5, 10, 10));
                //g.FillEllipse(new Pen(Color.Green).Brush, new RectangleF(end.X - 2, end.Y - 2, 4, 4));

            } while (nextdegreesidx <= 180 && segmentsdone != segments);
        }
        public PointF[] GetOvalPoints(float semi_major_axis, float semi_minor_axis)
        {
            const int points = 360;
            PointF[] ovalPoints = new PointF[points];
            for (int i = 0; i < points; i++)
            {
                double theta = (2 * Math.PI / points) * i; // Angle in radians
                float x = semi_major_axis * (float)Math.Cos(theta);
                float y = semi_minor_axis * (float)Math.Sin(theta);
                ovalPoints[i] = new PointF(x, y);
            }

            return ovalPoints;
        }
    }
}
