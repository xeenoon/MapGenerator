using System.Drawing.Imaging;

namespace TerrainGenerator
{

    public struct Bump
    {
        public double degrees;
        public double radius;

        public Bump(double degrees, double radius)
        {
            this.degrees = degrees;
            this.radius = radius;
        }
    }
    public class ProceduralShape
    {
        public List<Point> bounds = new List<Point>();
        public Rectangle rect_bounds;

        public List<Point> outer_bounds = new List<Point>();
        public List<Point> inner_bounds = new List<Point>();
        public Bitmap bakeddistances;
        public BitmapData bakeddistances_data;
        public Bitmap bakedbounds;
        public BitmapData bakedbounds_data;

        public Rectangle bakedrectangle;
        internal const int blenddst = 20;
        public static int max_blenddst = (int)Math.Sqrt(blenddst * blenddst * 2);
        const double MAX_BUMP_SIZE = 0.05; //Dont let a bump be more than 0.4x the 

        public ProceduralShape()
        {

        }
        public void GenerateShape(Rectangle bounds, int points, int seed = -1)
        {
            Random r = seed == -1 ? new Random() : new Random(seed); //Assign with seed if it is available, otherwise make it completely randmo
            //Generate some lakeish shape inside the bounds
            //int bumps = 2; //Lake by default is a circle, this changes the amount of bumps on the side of the lake
            int bumps = r.Next(8, 10);
            List<Bump> bumpdegrees = new List<Bump>();
            for (int i = 0; i < bumps; i++)
            {
                //Create n bumps inside the list at a random degrees
                double radius = r.NextDouble() * Math.Sqrt((bounds.Width * bounds.Height) * MAX_BUMP_SIZE); //Find the max area, then get the sqrt to find the max area
                bumpdegrees.Add(new Bump(r.NextDouble() * 360, radius));
            }

            for (double i = 0; i < 360; i += (360 / points))
            {
                double angleInRadians = i * (Math.PI / 180.0);

                //Ellipse equation is (x^2)/(a^2) + (y^2)/(b^2) = 1
                //The parametric equations for an ellipse are given by:

                //x(θ) = acos(θ)
                //y(θ) = bsin(θ)

                double x = (bounds.Width / 2) * Math.Cos(angleInRadians);
                double y = (bounds.Height / 2) * Math.Sin(angleInRadians);
                double radius = Math.Sqrt(x * x + y * y);

                //Check for bumps
                foreach (Bump b in bumpdegrees)
                {
                    var b_radians = b.degrees * (Math.PI / 180.0);
                    double distance = AngleCartesianDistance(angleInRadians, b_radians, bounds);
                    if (distance < b.radius * 4) //Inside the influence of the bump
                    {
                        //Imagine a new rectangle for a new ellipse to be drawn
                        //Rectangle will be on an angle
                        //First point of y distance radius, x distance radius
                        double maxratio = ((radius + b.radius) / radius);
                        PointF maxradius_point = new PointF((float)(maxratio * Math.Cos(b_radians) * (bounds.Width / 2)), (float)(maxratio * Math.Sin(b_radians) * (bounds.Height / 2)));


                        //Increase the new radius by an inverse scale of the bumps radius
                        //For distance = 0, radius = currentradius + b.radius
                        //For distance = b.radius, radius = currentradius



                        double newradius = radius + CurvePoints(distance, b.radius * 4, b.radius / 2);
                        double ratio = newradius / radius;
                        x *= ratio;
                        y *= ratio; //find the ratio difference between the radius's and multiply the new points by that value
                    }
                }


                this.bounds.Add(new Point((int)x + bounds.X + bounds.Width / 2, (int)y + bounds.Y + bounds.Height / 2)); //Adjust the points from relative to cartesian (0,0) to the box
            }
            int left = this.bounds.Min(p => p.X);
            int top = this.bounds.Min(p => p.Y);
            int width = this.bounds.Max(p => p.X) - left;
            int height = this.bounds.Max(p => p.Y) - top;
            rect_bounds = new Rectangle(left, top, width, height);


            bakeddistances = new Bitmap(rect_bounds.Width + max_blenddst * 2, rect_bounds.Height + max_blenddst * 2);
            bakedrectangle = new Rectangle(left - max_blenddst, top - max_blenddst, bakeddistances.Width, bakeddistances.Height);

            Graphics g = Graphics.FromImage(bakeddistances);
            List<Point[]> blendareas = new List<Point[]>();
            g.FillPolygon(new Pen(Color.FromArgb(255, 255, 255, 255)).Brush, this.bounds.ToArray().ScalePolygon(blenddst, new Point(-left + max_blenddst, -top + max_blenddst)));

            for (int scale = -blenddst; scale < blenddst; ++scale) //Draw from blenddst to -blenddst
            {
                //int scale = (2 * blenddst) - (i - blenddst);
                int distance = Math.Abs(scale);

                Color todraw = Color.FromArgb(255, distance, distance, distance); //Cache distance values in a bitmap
                g.FillPolygon(new Pen(todraw).Brush, this.bounds.ToArray().ScalePolygon(-scale, new Point(-left + max_blenddst, -top + max_blenddst)));
            }
            //g.FillPolygon(new Pen(Color.FromArgb(255, 255, 255, 255)).Brush, result.bounds.ToArray().ScalePolygon(-blenddst, new Point(-left + max_blenddst, -top + max_blenddst)));

            bakeddistances_data = bakeddistances.LockBits(new Rectangle(0, 0, bakeddistances.Width, bakeddistances.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppPArgb);

            bakedbounds = new Bitmap(rect_bounds.Width + max_blenddst * 2, rect_bounds.Height + max_blenddst * 2);
            g = Graphics.FromImage(bakedbounds);
            g.FillPolygon(new Pen(Color.FromArgb(255, 255, 0, 0)).Brush, this.bounds.ToArray().Offset(new Point(-left + max_blenddst, -top + max_blenddst))); //Mark the pixel
            bakedbounds_data = bakedbounds.LockBits(new Rectangle(0, 0, bakeddistances.Width, bakeddistances.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppPArgb);

        }
        public unsafe int DistanceTo(Point p)
        {
            if (bakedrectangle.Contains(p))
            {
                Point adjusted = new Point(p.X - bakedrectangle.Left, p.Y - bakedrectangle.Top);
                int checkidx = adjusted.X * 4 + adjusted.Y * bakeddistances_data.Stride;

                if (checkidx >= 0 && checkidx < bakeddistances_data.Stride * bakeddistances_data.Height)
                {
                    return ((byte*)bakeddistances_data.Scan0)[checkidx]; //BGRA
                }
                else
                {
                    throw new IndexOutOfRangeException("x or y out of range");
                }
            }

            return -1; //Outside bounds
        }
        public unsafe bool Contains(Point p)
        {
            if (bakedrectangle.Contains(p))
            {
                Point adjusted = new Point(p.X - bakedrectangle.Left, p.Y - bakedrectangle.Top);
                int checkidx = adjusted.X * 4 + adjusted.Y * bakeddistances_data.Stride;

                if (checkidx >= 0 && checkidx < bakeddistances_data.Stride * bakeddistances_data.Height)
                {
                    return ((byte*)bakedbounds_data.Scan0)[checkidx + 2] == 255;
                }
                else
                {
                    throw new IndexOutOfRangeException("x or y out of range");
                }
            }

            return false; //Outside bounds
        }
        public static double CurvePoints(double distance, double xcutoff, double ycutoff)
        {
            //distance = x, xcutoff = r, ycutoff = c
            //y = ((c) / (r ^ (2)))(-x ^ (2) + r ^ (2))
            return (ycutoff / (xcutoff * xcutoff)) * (-distance * distance + xcutoff * xcutoff);
        }
        private static double AngleCartesianDistance(double angleInRadians, double b_radians, Rectangle bounds)
        {
            return Math.Sqrt(Math.Pow((Math.Sin(b_radians) - Math.Sin(angleInRadians)) * bounds.Height, 2) + Math.Pow((Math.Cos(b_radians) - Math.Cos(angleInRadians)) * bounds.Width, 2));
        }
    }
}