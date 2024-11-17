namespace TerrainGenerator
{
    using System;
    using System.Drawing;
    using System.Drawing.Drawing2D;
    using System.Drawing.Imaging;

    public class CurvedBrick
    {
        public int ovalwidth;
        public int ovalheight;
        private PointF bottomleft;
        private PointF topleft;
        private PointF bottomright;
        private PointF topright;

        private PointF rotated_bottomleft;
        private PointF rotated_topleft;
        private PointF rotated_bottomright;
        private PointF rotated_topright;


        private PointF brickstart_bottomleft;
        private PointF brickstart_topleft;
        private PointF brickstart_bottomright;
        private PointF brickstart_topright;
        private PointF ovalcentre;
        public float width;

        private PointF start;
        private PointF end;
        private float rotationoffset;
        public static Bitmap darkstone;
        public static Bitmap lightstone;
        public static void Setup()
        {
            string exePath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            darkstone = (Bitmap)Image.FromFile(exePath + "\\images\\darkstone.jpg");
            lightstone = (Bitmap)Image.FromFile(exePath + "\\images\\smoothestone.png");
        }

        public CurvedBrick(int ovalwidth, int ovalheight, PointF start, PointF end, PointF circlecentre, float width, float rotationoffset)
        {
            this.start = start;
            this.end = end;
            this.ovalwidth = ovalwidth;
            this.ovalheight = ovalheight;
            this.ovalcentre = circlecentre;
            this.rotationoffset = rotationoffset;

            // Calculate the unit vectors from the circle center to the start and end points
            var radiusVectorStart = start.Subtract(circlecentre).UnitVector();
            var radiusVectorEnd = end.Subtract(circlecentre).UnitVector();

            // Now apply the width and calculate the bottomleft, topleft, bottomright, and topright for the unrotated version
            bottomleft = start.Subtract(radiusVectorStart.Scale(width / 2f));
            topleft = start.Add(radiusVectorStart.Scale(width / 2f));

            bottomright = end.Subtract(radiusVectorEnd.Scale(width / 2f));
            topright = end.Add(radiusVectorEnd.Scale(width / 2f));

            // Apply the rotation to each point around the circle center
            rotated_bottomleft = RotatePoint(bottomleft, circlecentre, rotationoffset);
            rotated_topleft = RotatePoint(topleft, circlecentre, rotationoffset);
            rotated_bottomright = RotatePoint(bottomright, circlecentre, rotationoffset);
            rotated_topright = RotatePoint(topright, circlecentre, rotationoffset);

            this.width = width;
        }

        // Helper method to rotate a point around a given center by a specified angle
        private PointF RotatePoint(PointF point, PointF center, float angle)
        {
            float cosAngle = (float)Math.Cos(angle);
            float sinAngle = (float)Math.Sin(angle);

            // Calculate the relative position of the point to the center
            float dx = point.X - center.X;
            float dy = point.Y - center.Y;

            // Apply the rotation matrix
            float rotatedX = center.X + (cosAngle * dx - sinAngle * dy);
            float rotatedY = center.Y + (sinAngle * dx + cosAngle * dy);

            return new PointF(rotatedX, rotatedY);
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
                int radius = 2;
                float diameter = radius * 2; // Calculate the diameter (2 * radius)

                //Draw main brick area
          //      g.FillEllipse(Brushes.Purple, topleft.X - radius, topleft.Y - radius, diameter, diameter);
          //      g.FillEllipse(Brushes.Purple, topright.X - radius, topright.Y - radius, diameter, diameter);
          //      g.FillEllipse(Brushes.Purple, bottomleft.X - radius, bottomleft.Y - radius, diameter, diameter);
          //      g.FillEllipse(Brushes.Purple, bottomright.X - radius, bottomright.Y - radius, diameter, diameter);
          //      g.FillEllipse(Brushes.Black, ovalcentre.X - radius*2, ovalcentre.Y - radius*2, diameter*2, diameter*2);

                var top = EdgePoints(g, topleft, topright, ovalwidth + width / 2, ovalheight + width / 2);
                var bottom = EdgePoints(g, bottomleft, bottomright, ovalwidth - width / 2, ovalheight - width / 2);
                bottom.Reverse();

                // Draw an ellipse at each specified point with the given radius
           //     g.FillEllipse(Brushes.Green, rotated_topleft.X - radius, rotated_topleft.Y - radius, diameter, diameter);
           //     g.FillEllipse(Brushes.Green, rotated_topright.X - radius, rotated_topright.Y - radius, diameter, diameter);
           //     g.FillEllipse(Brushes.Green, rotated_bottomleft.X - radius, rotated_bottomleft.Y - radius, diameter, diameter);
           //     g.FillEllipse(Brushes.Green, rotated_bottomright.X - radius, rotated_bottomright.Y - radius, diameter, diameter);
                foreach (var point in top.Concat(bottom))
                {
                   // g.FillEllipse(Brushes.Orange, point.X - radius, point.Y - radius, diameter, diameter);
                }
                //return;
                using (TextureBrush textureBrush = new TextureBrush(lightstone))
                {
                    var scaleMatrix = new Matrix();
                    scaleMatrix.Scale(0.2f, 0.2f); // Apply 0.1x scaling
                    textureBrush.Transform = scaleMatrix;

                    // Fill the polygon using the darkstone texture (with slight transparency)
                    g.FillPolygon(textureBrush, top.Concat(bottom).ToArray());
                }
                //Draw shadows
                Bitmap temp = new Bitmap(display.Width, display.Height);
                Graphics tempgraphics = Graphics.FromImage(temp);
                tempgraphics.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                float thikkness = width / 10f;

                for (float dst = 0; dst <= thikkness; ++dst) //Use float so /2 doesn't round
                {
                    double startangle = Math.Atan2(start.Y - ovalcentre.Y, start.X - ovalcentre.X);
                    double start_angleChange = startangle + CalculateAngleForDistance(start, ovalcentre, dst / 2, ovalwidth - dst, ovalheight - dst, true); //Hack, check angle positions
                    float magnitude = start.Subtract(ovalcentre).Magnitude();
                    PointF newstart = ovalcentre.Add(new PointF((float)Math.Cos(start_angleChange) * magnitude, (float)Math.Sin(start_angleChange) * magnitude));
                    var radiusvector = newstart.Subtract(ovalcentre).UnitVector();
                    brickstart_bottomleft = newstart.Subtract(radiusvector.Scale((width) / 2 - dst));

                    start_angleChange = startangle + CalculateAngleForDistance(start, ovalcentre, dst / 2, ovalwidth + dst, ovalheight + dst, true); //Hack, check angle positions
                    newstart = ovalcentre.Add(new PointF((float)Math.Cos(start_angleChange) * magnitude, (float)Math.Sin(start_angleChange) * magnitude));
                    brickstart_topleft = newstart.Add(radiusvector.Scale((width) / 2 - dst));

                    double endangle = Math.Atan2(end.Y - ovalcentre.Y, end.X - ovalcentre.X);
                    double end_angleChange = endangle + CalculateAngleForDistance(end, ovalcentre, dst / 2, ovalwidth - dst, ovalheight - dst, false); //Hack, check angle positions
                    magnitude = end.Subtract(ovalcentre).Magnitude();
                    PointF newend = ovalcentre.Add(new PointF((float)Math.Cos(end_angleChange) * magnitude, (float)Math.Sin(end_angleChange) * magnitude));
                    radiusvector = newend.Subtract(ovalcentre).UnitVector();
                    brickstart_bottomright = newend.Subtract(radiusvector.Scale((width) / 2 - dst));

                    end_angleChange = endangle + CalculateAngleForDistance(end, ovalcentre, dst / 2, ovalwidth + dst, ovalheight + dst, false); //Hack, check angle positions
                    newend = ovalcentre.Add(new PointF((float)Math.Cos(end_angleChange) * magnitude, (float)Math.Sin(end_angleChange) * magnitude));
                    brickstart_topright = newend.Add(radiusvector.Scale((width) / 2 - dst));

                    //alpha should be 255 when dst = 0 and 0 when dst = thikkness
                    var alpha = (int)(125 * (1 - (dst / thikkness)));
                    var bricktop = EdgePoints(g, brickstart_topleft, brickstart_topright, ovalwidth + width / 2 - dst * 2, ovalheight + width / 2 - dst * 2);
                    var brickbottom = EdgePoints(g, brickstart_bottomleft, brickstart_bottomright, ovalwidth - width / 2 + dst * 2, ovalheight - width / 2 + dst * 2);
                    brickbottom.Reverse();

                    var rotated_brickstart_bottomleft = RotatePoint(brickstart_bottomleft, ovalcentre, rotationoffset);
                    var rotated_brickstart_topleft = RotatePoint(brickstart_topleft, ovalcentre, rotationoffset);
                    var rotated_brickstart_bottomright = RotatePoint(brickstart_bottomright, ovalcentre, rotationoffset);
                    var rotated_brickstart_topright = RotatePoint(brickstart_topright, ovalcentre, rotationoffset);

               //     g.FillEllipse(Brushes.Red, rotated_brickstart_bottomleft.X - radius, rotated_brickstart_bottomleft.Y - radius, diameter, diameter);
               //     g.FillEllipse(Brushes.Red, rotated_brickstart_topleft.X - radius, rotated_brickstart_topleft.Y - radius, diameter, diameter);
               //     g.FillEllipse(Brushes.Red, rotated_brickstart_bottomright.X - radius, rotated_brickstart_bottomright.Y - radius, diameter, diameter);
               //     g.FillEllipse(Brushes.Red, rotated_brickstart_topright.X - radius, rotated_brickstart_topright.Y - radius, diameter, diameter);


                    tempgraphics.FillPolygon(new Pen(Color.FromArgb(alpha, 0, 0, 0)).Brush, bricktop.Concat(brickbottom).ToArray());

                    foreach (var point in bricktop.Concat(brickbottom))
                    {
                     //   g.FillEllipse(Brushes.Orange, point.X - radius, point.Y - radius, diameter, diameter);
                    }
                }
                g.DrawImage(temp, new Point(0, 0));
            }
        }

        private List<PointF> EdgePoints(Graphics g, PointF start, PointF end, float ovalWidth, float ovalHeight)
        {
            List<PointF> points = new List<PointF>();
            // Convert rotation angle to radians

            // Measure the angle difference
            float startAngle = (float)Math.Atan2(start.Y - ovalcentre.Y, start.X - ovalcentre.X);
            float endAngle = (float)Math.Atan2(end.Y - ovalcentre.Y, end.X - ovalcentre.X);


            // Ensure angles are ordered correctly
            if (startAngle > endAngle) endAngle += 2 * (float)Math.PI;

            const int segments = 25; // Number of segments
            float angleStep = (endAngle - startAngle) / (float)segments;

            for (int i = 0; i <= segments; i++)
            {
                float angle = startAngle + i * angleStep;

                // Parametric equations for the ellipse
                float a = ovalWidth;  // Semi-major axis
                float b = ovalHeight; // Semi-minor axis

                float unrotatedX = a * b * (float)Math.Cos(angle) /
                                   (float)Math.Sqrt((b * b * Math.Cos(angle) * Math.Cos(angle)) +
                                                    (a * a * Math.Sin(angle) * Math.Sin(angle)));
                float unrotatedY = a * b * (float)Math.Sin(angle) /
                                   (float)Math.Sqrt((b * b * Math.Cos(angle) * Math.Cos(angle)) +
                                                    (a * a * Math.Sin(angle) * Math.Sin(angle)));
                points.Add(RotatePoint(new PointF(unrotatedX, unrotatedY).Add(ovalcentre), ovalcentre, rotationoffset));
            }

            return points;
        }

    }
}