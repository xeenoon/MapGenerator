using System.Diagnostics;

namespace TerrainGenerator
{
    class Caterpillar
    {
        public List<PointF> spine = new List<PointF>();
        public double[] sintimes;
        public static int pointsperbump = 8;
        public static int sectionwidth = 50;
        public static int sectionheight = 50;
        public Caterpillar(int length, Point head)
        {
            if (length <= 2)
            {
                throw new Exception("Length must be over 3");
            }
            sintimes = new double[length];
            for (int i = 0; i < length; ++i)
            {
                sintimes[i] = (i % pointsperbump) * Math.PI / 4;
                spine.Add(new PointF(head.X - sectionwidth * i, head.Y + (int)(Math.Sin(sintimes[i]) * sectionheight / 2)));
            }
        }
        public void MoveTowards(Point p)
        {
            //Assume it will be called once a frame
            const double speed = 2;

            //Find the angle between the head and the neck
            var head = spine[0];
            var neck = spine[1];

            //Rotate the head towards the mouse
            const double rotationspeed = 0.05;
            int lastpoint = spine.Count()-1;
            spine[0] = RotateTowards(spine[1], spine[0], p, rotationspeed);
            spine[0] = DragPoint(spine[0], CalculateAngle(spine[1], spine[0]), speed);

            //Recursively go through all spine points, moving it towards the LAST one
            for(int i = 1; i < spine.Count(); ++i)
            {
                var me = spine[i];
                var infront = spine[i-1];
                double angle = CalculateAngle(me, infront);

                //Drag myself towards this new angle by speed
                spine[i] = DragPoint(spine[i], angle, speed);
            }

        }
        private static PointF DragPoint(PointF p, double angle, double speed)
        {
            PointF dp = new PointF((float)(Math.Cos(angle) * speed),(float)(Math.Sin(angle) * speed));
            return new PointF(p.X + dp.X, p.Y + dp.Y);
        }
        public static PointF RotateTowards(PointF p0, PointF p1, PointF p2, double rotationspeed)
        {
            double headangle = CalculateAngle(p0, p1);
            double mouseangle = CalculateAngle(p0, p2);

            double newangle;
            if (Math.Abs(mouseangle - headangle) < rotationspeed)
            {
                newangle = mouseangle;
            }
            else
            {
                double difference = mouseangle - headangle;

                if (difference > Math.PI) difference -= 2 * Math.PI;
                else if (difference < -Math.PI) difference += 2 * Math.PI;

                newangle = headangle + (difference > 0 ? rotationspeed : -rotationspeed);
            }
            return new PointF((float)(Math.Cos(newangle) * p0.DistanceTo(p1)) + p0.X, (float)(Math.Sin(newangle) * p0.DistanceTo(p1) + p0.Y));
        }
        public static double CalculateAngle(PointF p1, PointF p2)
        {
            double deltaY = p2.Y - p1.Y;
            double deltaX = p2.X - p1.X;
            double angle = Math.Atan2(deltaY, deltaX);

            // Adjust angle to be between 0 and 360 degrees
            if (angle < 0)
            {
                angle += Math.PI * 2;
            }
            return angle;
        }
        public void Draw(Bitmap result)
        {
            Graphics g = Graphics.FromImage(result);
            foreach (var p in spine)
            {
                int size = 10;
                g.FillEllipse(new Pen(Color.Red).Brush, new RectangleF(p.X - size / 2, p.Y - size / 2, size, size));
            }
        }
    }
}