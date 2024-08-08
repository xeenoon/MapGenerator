using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace TerrainGenerator
{
    public class Puddle
    {
        public List<Point> bounds = new List<Point>();
        public Rectangle rect_bounds;

        public List<Point> outer_bounds = new List<Point>();
        public List<Point> inner_bounds = new List<Point>();
        public Bitmap bakeddistances;
        public BitmapData bakeddistances_data;

        public Rectangle bakedrectangle;
        public Puddle()
        {

        }
        const double MAX_BUMP_SIZE = 0.05; //Dont let a bump be more than 0.4x the 
        public static Puddle GeneratePuddle(Rectangle bounds, int points, int seed = -1)
        {
            Puddle result = new Puddle();
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



                        double newradius = radius + PuddleSmootheCurve(distance, b.radius * 4, b.radius / 2);
                        double ratio = newradius / radius;
                        x *= ratio;
                        y *= ratio; //find the ratio difference between the radius's and multiply the new points by that value
                    }
                }


                result.bounds.Add(new Point((int)x + bounds.X + bounds.Width / 2, (int)y + bounds.Y + bounds.Height / 2)); //Adjust the points from relative to cartesian (0,0) to the box
            }
            int left = result.bounds.Min(p => p.X);
            int top = result.bounds.Min(p => p.Y);
            int width = result.bounds.Max(p => p.X) - left;
            int height = result.bounds.Max(p => p.Y) - top;
            result.rect_bounds = new Rectangle(left, top, width, height);


            result.bakeddistances = new Bitmap(result.rect_bounds.Width + max_blenddst * 2, result.rect_bounds.Height + max_blenddst * 2);
            result.bakedrectangle = new Rectangle(left - max_blenddst, top - max_blenddst, result.bakeddistances.Width, result.bakeddistances.Height);

            Graphics g = Graphics.FromImage(result.bakeddistances);
            List<Point[]> blendareas = new List<Point[]>();
            g.FillPolygon(new Pen(Color.FromArgb(255, 255, 255, 255)).Brush, result.bounds.ToArray().ScalePolygon(blenddst, new Point(-left + max_blenddst, -top + max_blenddst)));

            for (int scale = -blenddst; scale < blenddst; ++scale) //Draw from blenddst to -blenddst
            {
                //int scale = (2 * blenddst) - (i - blenddst);
                int distance = Math.Abs(scale);

                Color todraw = Color.FromArgb(255, distance, distance, distance); //Cache distance values in a bitmap
                g.FillPolygon(new Pen(todraw).Brush, result.bounds.ToArray().ScalePolygon(-scale, new Point(-left + max_blenddst, -top + max_blenddst)));
            }
            //g.FillPolygon(new Pen(Color.FromArgb(255, 255, 255, 255)).Brush, result.bounds.ToArray().ScalePolygon(-blenddst, new Point(-left + max_blenddst, -top + max_blenddst)));

            result.bakeddistances_data = result.bakeddistances.LockBits(new Rectangle(0, 0, result.bakeddistances.Width, result.bakeddistances.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppPArgb);

            return result;
        }
        public static double PuddleSmootheCurve(double distance, double xcutoff, double ycutoff)
        {
            //distance = x, xcutoff = r, ycutoff = c
            //y = ((c) / (r ^ (2)))(-x ^ (2) + r ^ (2))
            return (ycutoff / (xcutoff * xcutoff)) * (-distance * distance + xcutoff * xcutoff);
        }
        public unsafe int DistanceTo(Point p)
        {
            if (bakedrectangle.Contains(p))
            {
                Point adjusted = new Point(p.X - bakedrectangle.Left, p.Y - bakedrectangle.Top);
                int checkidx = adjusted.X * 4 + adjusted.Y * bakeddistances_data.Stride;
                return ((byte*)bakeddistances_data.Scan0)[checkidx]; //BGRA
            }
            return -1; //Outside bounds
        }

        private static double AngleCartesianDistance(double angleInRadians, double b_radians, Rectangle bounds)
        {
            return Math.Sqrt(Math.Pow((Math.Sin(b_radians) - Math.Sin(angleInRadians)) * bounds.Height, 2) + Math.Pow((Math.Cos(b_radians) - Math.Cos(angleInRadians)) * bounds.Width, 2));
        }
        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe void* memcpy(void* dest, void* src, UIntPtr count);
        const int blenddst = 50;
        public static int max_blenddst = (int)Math.Sqrt(blenddst * blenddst * 2);
        public unsafe void DrawPuddle(Bitmap result)
        {

            Bitmap polygonmarker = new Bitmap(result.Width, result.Height);
            Graphics graphics = Graphics.FromImage(polygonmarker);
            graphics.FillPolygon(new Pen(Color.FromArgb(255, 255, 0, 0)).Brush, bounds.ToArray()); //Mark the pixel

            var markdata = polygonmarker.LockBits(new Rectangle(0, 0, polygonmarker.Width, polygonmarker.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppPArgb);
            var markptr = (byte*)markdata.Scan0;

            BitmapData write = result.LockBits(new Rectangle(0, 0, result.Width, result.Height), ImageLockMode.ReadWrite, PixelFormat.Format32bppPArgb);

            string exePath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            var mudbmp = (Bitmap)Image.FromFile(exePath + "\\images\\dirtseam.jpg");
            var muddata = mudbmp.LockBits(new Rectangle(0, 0, mudbmp.Width, mudbmp.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppPArgb);

            const int watertilefactor = 8;
            var waterbmp = (Bitmap)Image.FromFile(exePath + "\\images\\waterseam.jpg");
            var waterdata = waterbmp.LockBits(new Rectangle(0, 0, waterbmp.Width, waterbmp.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppPArgb);


            byte* resultptr = (byte*)write.Scan0;
            byte* mudptr = (byte*)muddata.Scan0;
            byte* waterptr = (byte*)waterdata.Scan0;

            Point topleft = new Point(bounds.Min(p => p.X), bounds.Min(p => p.Y));
            Point botright = new Point(bounds.Max(p => p.X), bounds.Max(p => p.Y));

            for (int x = topleft.X; x < botright.X; ++x)
            {
                for (int y = topleft.Y; y < botright.Y; ++y)
                {
                    byte* result_loc = resultptr + x * 4 + y * 4 * result.Width;
                    byte* mud_loc = mudptr + (x % mudbmp.Width) * 4 + (y % mudbmp.Height) * 4 * mudbmp.Width;
                    byte* water_loc = waterptr + ((x * watertilefactor) % waterbmp.Width) * 4 + ((y * watertilefactor) % waterbmp.Height) * 4 * waterbmp.Width;

                    const double waterblend = 0.5;
                    if (markptr[x * 4 + y * 4 * result.Width + 2] == 255) // Am I in the polygon?
                    {
                        //var distance = (double)DistanceTo(new Point(x, y));
                        var distance = bounds.ToArray().DistanceFromPointToPolygon(new Point(x, y));
                        if (distance <= blenddst && distance != -1) //Blend on edges
                        {
                            //double blendfactor = distance / (blenddst * blendstrength) + (1 - (1.0 / blendstrength));
                            double blendfactor = Math.Min(distance / blenddst, 1);

                            Extensions.BlendColors(result_loc, mud_loc, blendfactor);
                            Extensions.BlendColors(result_loc, water_loc, blendfactor * waterblend);
                        }
                        else
                        {
                            memcpy(result_loc, mud_loc, 3);
                            Extensions.BlendColors(result_loc, water_loc, waterblend);

                        }

                        // Set alpha to 255
                        result_loc[3] = 255;
                    }
                }
            }
            result.UnlockBits(write);
            mudbmp.UnlockBits(muddata);
            polygonmarker.UnlockBits(markdata);
        }
    }
}