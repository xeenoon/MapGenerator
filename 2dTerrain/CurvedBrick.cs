namespace TerrainGenerator
{
    using System;
    using System.Drawing;

    public class CurvedBrick
    {
        public int xstretch;
        public int ystretch;
        private PointF bottomleft;
        private PointF topleft;
        private PointF bottomright;
        private PointF topright;
        private PointF circlecentre;
        public float width;

        public CurvedBrick(int xstretch, int ystretch, PointF start, PointF end, PointF circlecentre, float width)
        {
            this.xstretch = xstretch;
            this.ystretch = ystretch;
            this.circlecentre = circlecentre;

            // Assume start and end are on the circle
            var radiusvector = start.Subtract(circlecentre).UnitVector();
            bottomleft = start.Subtract(radiusvector.Scale(width / 2f));
            topleft = start.Add(radiusvector.Scale(width / 2f));

            radiusvector = end.Subtract(circlecentre).UnitVector();

            bottomright = end.Subtract(radiusvector.Scale(width / 2f));
            topright = end.Add(radiusvector.Scale(width / 2f));
            this.width = width;
        }

        public void Draw(Bitmap display)
        {
            using (Graphics g = Graphics.FromImage(display))
            {
                g.Clear(Color.White);

                // Draw curved edges
                DrawCurvedEdge(g, topleft, topright);
                DrawCurvedEdge(g, bottomleft, bottomright);

                // Draw straight vertical lines
                DrawStraightLine(g, topleft, bottomleft);
                DrawStraightLine(g, topright, bottomright);
            }
        }

        private void DrawCurvedEdge(Graphics g, PointF start, PointF end)
        {
            var magnitude = start.Subtract(circlecentre).Magnitude();

            // Measure the angle difference
            float startAngle = (float)Math.Atan2(start.Y - circlecentre.Y, start.X - circlecentre.X);
            float endAngle = (float)Math.Atan2(end.Y - circlecentre.Y, end.X - circlecentre.X);

            // Ensure angles are ordered correctly
            if (startAngle > endAngle) endAngle += 2 * (float)Math.PI;

            const int segments = 5; // Number of segments
            float angleStep = (endAngle - startAngle) / segments;

            PointF[] points = new PointF[segments + 1];

            for (int i = 0; i <= segments; i++)
            {
                float angle = startAngle + i * angleStep;
                float curveX = circlecentre.X + (float)(Math.Cos(angle) * magnitude);
                float curveY = circlecentre.Y + (float)(Math.Sin(angle) * magnitude);
                points[i] = new PointF(curveX, curveY);
            }

            // Draw line segments between computed points
            for (int i = 0; i < points.Length - 1; i++)
            {
                g.DrawLine(Pens.Blue, points[i], points[i + 1]);
            }
        }

        private void DrawStraightLine(Graphics g, PointF start, PointF end)
        {
            g.DrawLine(Pens.Red, start, end);
        }
    }
}