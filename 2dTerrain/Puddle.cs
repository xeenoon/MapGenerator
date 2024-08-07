using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace TerrainGenerator
{
    public class Puddle
    {
        public List<Point> bounds = new List<Point>();
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
            return result;
        }
        public static double PuddleSmootheCurve(double distance, double xcutoff, double ycutoff)
        {
            //distance = x, xcutoff = r, ycutoff = c
            //y = ((c) / (r ^ (2)))(-x ^ (2) + r ^ (2))
            return (ycutoff / (xcutoff * xcutoff)) * (-distance * distance + xcutoff * xcutoff);
        }

        private static double AngleCartesianDistance(double angleInRadians, double b_radians, Rectangle bounds)
        {
            return Math.Sqrt(Math.Pow((Math.Sin(b_radians) - Math.Sin(angleInRadians)) * bounds.Height, 2) + Math.Pow((Math.Cos(b_radians) - Math.Cos(angleInRadians)) * bounds.Width, 2));
        }
        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe void* memcpy(void* dest, void* src, UIntPtr count);

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

                    const double blenddst = 50;
                    const double waterblend = 0.5;
                    if (markptr[x * 4 + y * 4 * result.Width + 2] == 255) // Am I in the polygon?
                    {
                        var distance = bounds.ToArray().DistanceFromPointToPolygon(new Point(x, y));
                        if (distance <= blenddst) //Blend on edges
                        {
                            const double blendstrength = 3;
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