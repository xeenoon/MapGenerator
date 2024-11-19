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
            float angleIncrement = (float)(2 * Math.PI / segments);
            DrawSide(result, centre, semi_major_axis, semi_minor_axis, rotationOffset, angleIncrement);

            var adjustedcentre = centre.Subtract(end.Subtract(start).Perpendicular().UnitVector().Scale(walkwaywidth / 4));
            semi_major_axis += walkwaywidth * 2;
            semi_minor_axis += walkwaywidth * 2;
            DrawSide(result, centre, semi_major_axis, semi_minor_axis, rotationOffset, angleIncrement);

        }

        private void DrawSide(Bitmap result, PointF centre, float semi_major_axis, float semi_minor_axis, float rotationOffset, float angleIncrement)
        {
            // Draw the curved bricks along the rotated oval
            for (double angle = 0; angle < Math.PI; angle += angleIncrement)
            {
                // Calculate the start and end positions based on the angle
                PointF start = centre.Add(new PointF((float)Math.Cos(angle) * semi_major_axis, (float)Math.Sin(angle) * semi_minor_axis));
                PointF end = centre.Add(new PointF((float)Math.Cos(angle + angleIncrement) * semi_major_axis, (float)Math.Sin(angle + angleIncrement) * semi_minor_axis));

                // Create and draw the curved brick using the CurvedBrick constructor
                CurvedBrick curvedBrick = new(
                    (int)semi_major_axis,
                    (int)semi_minor_axis,
                    start,
                    end,
                    centre,
                    brickwidth * 2,
                    rotationOffset
                );
                curvedBrick.Draw(result);  // Draw the curved brick on the result bitmap
            }
        }
    }
}
