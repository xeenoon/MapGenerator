namespace TerrainGenerator
{
    using System;
    using System.Drawing;

    public class CurvedBrick
    {
        public int ovalwidth;
        public int ovalheight;
        private PointF bottomleft;
        private PointF topleft;
        private PointF bottomright;
        private PointF topright;

        private PointF brickstart_bottomleft;
        private PointF brickstart_topleft;
        private PointF brickstart_bottomright;
        private PointF brickstart_topright;
        private PointF ovalcentre;
        public float width;
        public static float thikkness = 5;

        private PointF start;
        public static Bitmap darkstone;
        public static Bitmap lightstone;
        public static void Setup()
        {
            string exePath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            darkstone = (Bitmap)Image.FromFile(exePath + "\\images\\darkstone.jpg");
            lightstone = (Bitmap)Image.FromFile(exePath + "\\images\\stone.png");
        }

        public CurvedBrick(int ovalwidth, int ovalheight, PointF start, PointF end, PointF circlecentre, float width)
        {
            this.start = start;
            this.ovalwidth = ovalwidth;
            this.ovalheight = ovalheight;
            this.ovalcentre = circlecentre;

            // Assume start and end are on the circle
            var radiusvector = start.Subtract(circlecentre).UnitVector();
            bottomleft = start.Subtract(radiusvector.Scale(width / 2f));
            topleft = start.Add(radiusvector.Scale(width / 2f));

            radiusvector = end.Subtract(circlecentre).UnitVector();
            bottomright = end.Subtract(radiusvector.Scale(width / 2f));
            topright = end.Add(radiusvector.Scale(width / 2f));

            double startangle = Math.Atan2(start.Y - circlecentre.Y, start.X - circlecentre.X);
            double start_angleChange = startangle + CalculateAngleForDistance(start, circlecentre, thikkness/2, ovalwidth - thikkness, ovalheight - thikkness, true); //Hack, check angle positions
            float magnitude = start.Subtract(circlecentre).Magnitude();
            PointF newstart = circlecentre.Add(new PointF((float)Math.Cos(start_angleChange) * magnitude, (float)Math.Sin(start_angleChange) * magnitude));
            radiusvector = newstart.Subtract(circlecentre).UnitVector();
            brickstart_bottomleft = newstart.Subtract(radiusvector.Scale((width) / 2 - thikkness));

            start_angleChange = startangle + CalculateAngleForDistance(start, circlecentre, thikkness/2, ovalwidth + thikkness, ovalheight + thikkness, true); //Hack, check angle positions
            newstart = circlecentre.Add(new PointF((float)Math.Cos(start_angleChange) * magnitude, (float)Math.Sin(start_angleChange) * magnitude));
            brickstart_topleft = newstart.Add(radiusvector.Scale((width) / 2 - thikkness));

            double endangle = Math.Atan2(end.Y - circlecentre.Y, end.X - circlecentre.X);
            double end_angleChange = endangle + CalculateAngleForDistance(end, circlecentre, thikkness/2, ovalwidth - thikkness, ovalheight - thikkness, false); //Hack, check angle positions
            magnitude = end.Subtract(circlecentre).Magnitude();
            PointF newend = circlecentre.Add(new PointF((float)Math.Cos(end_angleChange) * magnitude, (float)Math.Sin(end_angleChange) * magnitude));
            radiusvector = newend.Subtract(circlecentre).UnitVector();
            brickstart_bottomright = newend.Subtract(radiusvector.Scale((width) / 2 - thikkness));

            end_angleChange = endangle + CalculateAngleForDistance(end, circlecentre, thikkness/2, ovalwidth + thikkness, ovalheight + thikkness, false); //Hack, check angle positions
            newend = circlecentre.Add(new PointF((float)Math.Cos(end_angleChange) * magnitude, (float)Math.Sin(end_angleChange) * magnitude));
            brickstart_topright = newend.Add(radiusvector.Scale((width) / 2 - thikkness));

            this.width = width;
        }
        public static double CalculateAngleForDistance(PointF start, PointF circleCentre, double distance, double width, double height, bool clockwise)
        {
            // Calculate the angle of the starting point relative to the ellipse's center
            float deltaX = start.X - circleCentre.X;
            float deltaY = start.Y - circleCentre.Y;
            double startAngle = Math.Atan2(deltaY * (width / height), deltaX); // Adjusted for ellipse scaling

            // Calculate the local radius (distance from center to the ellipse edge) at the given angle
            double radius = (width / 2) * (height / 2) /
                            Math.Sqrt(
                                Math.Pow((height / 2) * Math.Cos(startAngle), 2) +
                                Math.Pow((width / 2) * Math.Sin(startAngle), 2)
                            );

            // Calculate the angle change required to cover the specified distance
            double angleChange = distance / radius; // Arc length formula

            // Adjust the direction of the angle change based on the 'clockwise' parameter
            if (!clockwise)
            {
                angleChange = -angleChange; // Reverse the direction for counterclockwise
            }

            return angleChange;
        }

        public void Draw(Bitmap display)
        {
            using (Graphics g = Graphics.FromImage(display))
            {
                // Draw curved edges
                int radius = 5;

                var top = EdgePoints(g, topleft, topright, ovalwidth + width / 2, ovalheight + width / 2);
                var bottom = EdgePoints(g, bottomleft, bottomright, ovalwidth - width / 2, ovalheight - width / 2);
                bottom.Reverse();
                g.FillPolygon(new Pen(Color.DarkGray).Brush, top.Concat(bottom).ToArray());


                var bricktop = EdgePoints(g, brickstart_topleft, brickstart_topright, ovalwidth + width/2 - thikkness, ovalheight + width/2 - thikkness);
                var brickbottom = EdgePoints(g, brickstart_bottomleft, brickstart_bottomright, ovalwidth - width/2 + thikkness, ovalheight - width/2 + thikkness);
                brickbottom.Reverse();
                g.FillPolygon(new Pen(Color.LightGray).Brush, bricktop.Concat(brickbottom).ToArray());

                foreach (PointF p in bricktop.Concat(brickbottom))
                {
                    radius = 3;
                    //g.FillEllipse(new Pen(Color.Black).Brush, p.X - radius, p.Y - radius, radius * 2, radius * 2);
                }
            }
        }

        private List<PointF> EdgePoints(Graphics g, PointF start, PointF end, float ovalwidth, float ovalheight)
        {
            List<PointF> points = new List<PointF>();
            var magnitude = start.Subtract(ovalcentre).Magnitude();

            // Measure the angle difference
            float startAngle = (float)Math.Atan2(start.Y - ovalcentre.Y, start.X - ovalcentre.X);
            float endAngle = (float)Math.Atan2(end.Y - ovalcentre.Y, end.X - ovalcentre.X);


            // Ensure angles are ordered correctly
            if (startAngle > endAngle) endAngle += 2 * (float)Math.PI;

            const int segments = 5; // Number of segments
            float angleStep = (endAngle - startAngle) / (float)segments;

            for (int i = 0; i <= segments; i++)
            {
                float angle = startAngle + i * angleStep;

                // Parametric equations for the ellipse
                float a = ovalwidth; // Semi-major axis
                float b = ovalheight; // Semi-minor axis

                // Correctly calculate the intersection point on the ellipse's edge
                float curveX = ovalcentre.X + a * b * (float)Math.Cos(angle) /
                               (float)Math.Sqrt((b * b * Math.Cos(angle) * Math.Cos(angle)) +
                                                (a * a * Math.Sin(angle) * Math.Sin(angle)));

                float curveY = ovalcentre.Y + a * b * (float)Math.Sin(angle) /
                               (float)Math.Sqrt((b * b * Math.Cos(angle) * Math.Cos(angle)) +
                                                (a * a * Math.Sin(angle) * Math.Sin(angle)));

                points.Add(new PointF(curveX, curveY));
            }

            return points;
        }
    }
}