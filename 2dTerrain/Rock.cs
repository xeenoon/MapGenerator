using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Security.Cryptography.Xml;
using System.Text;
using System.Threading.Tasks;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;
using System.Timers;
using System.Threading;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace TerrainGenerator
{
    public class Rock
    {
        public List<Point> bounds = new List<Point>();
        public Rock()
        {

        }
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
        const double MAX_BUMP_SIZE = 0.1; //Dont let a bump be more than 0.4x the 

        public static Rock GenerateRock(Rectangle bounds, int points, int seed = -1)
        {
            Rock result = new Rock();
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



                        double newradius = radius + RockSmootheCurve(distance, b.radius * 4, b.radius / 2);
                        double ratio = newradius / radius;
                        x *= ratio;
                        y *= ratio; //find the ratio difference between the radius's and multiply the new points by that value
                    }
                }

                result.bounds.Add(new Point((int)x + bounds.X + bounds.Width / 2, (int)y + bounds.Y + bounds.Height / 2)); //Adjust the points from relative to cartesian (0,0) to the box
            }
            return result;
        }
        public static double RockSmootheCurve(double distance, double xcutoff, double ycutoff)
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
        public static extern IntPtr memcpy(IntPtr dest, IntPtr src, UIntPtr count);

        public unsafe void Draw(Bitmap image)
        {
            Graphics graphics = Graphics.FromImage(image);
            graphics.FillPolygon(new Pen(Color.FromArgb(255, 255, 0, 0)).Brush, bounds.ToArray()); //Mark the pixel

            //Find a bounds around the bounds of the rock
            Point topleft = new Point(bounds.OrderBy(p => p.X).FirstOrDefault().X, bounds.OrderBy(p => p.Y).FirstOrDefault().Y);
            Point bottomright = new Point(bounds.OrderByDescending(p => p.X).FirstOrDefault().X, bounds.OrderByDescending(p => p.Y).FirstOrDefault().Y);

            var resultbmp = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), System.Drawing.Imaging.ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format32bppRgb);
            var resultbmp_scan0 = (byte*)resultbmp.Scan0;

            var rock = (Bitmap)Image.FromFile("C:\\Users\\ccw10\\Downloads\\stoneseam.png");
            var rockbmp = rock.LockBits(new Rectangle(0, 0, rock.Width, rock.Height), System.Drawing.Imaging.ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format32bppRgb);
            var rockbmp_scan0 = (byte*)rockbmp.Scan0;

            const int shadowdst = 20;
            const int shinedst = 40;
            Point rockcentre = PolygonCentre(bounds.ToArray());

            for (int x = topleft.X - shadowdst; x < bottomright.X + shadowdst; ++x)
            {
                for (int y = topleft.Y - shadowdst; y < bottomright.Y + shadowdst; ++y)
                {
                    if (x >= image.Width || y >= image.Height || x < 0 || y < 0)
                    {
                        continue;
                    }
                    const int BYTES_PER_PIXEL = 4;
                    if (resultbmp_scan0[x * BYTES_PER_PIXEL + y * resultbmp.Stride + 2] == 255) // Am I in the polygon?
                    {
                        int resultIndex = x * BYTES_PER_PIXEL + y * resultbmp.Stride;
                        int rockIndex = (x % rock.Width) * BYTES_PER_PIXEL + (y % rock.Height) * rockbmp.Stride;

                        // Copy RGBA values from rockbmp_scan0 to resultbmp_scan0
                        memcpy((IntPtr)(resultbmp_scan0 + resultIndex), (IntPtr)(rockbmp_scan0 + rockIndex), BYTES_PER_PIXEL);

                        double centredst = rockcentre.DistanceTo(new Point(x, y));
                        if (centredst <= shinedst) //Lighten in centre
                        {
                            byte b = resultbmp_scan0[resultIndex];
                            byte g = resultbmp_scan0[resultIndex + 1];
                            byte r = resultbmp_scan0[resultIndex + 2];

                            const double shinestrength = 4;
                            //Should be equal to 1.5 where distance is 0
                            //Should be equal to 1 where distance is shinedst
                            //double shadowFactor = centredst/(double)shinedst;
                            double shadowFactor = 1 + ((1.0/shinestrength) * (1 - (centredst / shinedst)));
                            //double shadowFactor = centredst / (shadowdst * 3) + (1 - (1.0 / 3));
                            
                            resultbmp_scan0[resultIndex]     = (byte)Math.Min(b * shadowFactor, 255);
                            resultbmp_scan0[resultIndex + 1] = (byte)Math.Min(g * shadowFactor,255);
                            resultbmp_scan0[resultIndex + 2] = (byte)Math.Min(r * shadowFactor, 255);
                        }
                        // Set alpha to 255
                        resultbmp_scan0[resultIndex + 3] = 255;
                    }

                    var distance = DistanceFromPointToPolygon(new Point(x, y), bounds.ToArray());
                    if (distance <= shadowdst) //Darken on edges
                    {
                        int resultIndex = x * BYTES_PER_PIXEL + y * resultbmp.Stride;

                        byte b = resultbmp_scan0[resultIndex];
                        byte g = resultbmp_scan0[resultIndex + 1];
                        byte r = resultbmp_scan0[resultIndex + 2];

                        const double shadowstrength = 3;
                        double shadowFactor = distance / (shadowdst * shadowstrength) + (1 - (1.0 / shadowstrength));

                        resultbmp_scan0[resultIndex] = (byte)(b * shadowFactor);
                        resultbmp_scan0[resultIndex + 1] = (byte)(g * shadowFactor);
                        resultbmp_scan0[resultIndex + 2] = (byte)(r * shadowFactor);

                        // Set alpha to 255
                        resultbmp_scan0[resultIndex + 3] = 255;
                    }
                }
            }
            image.UnlockBits(resultbmp);
            rock.UnlockBits(rockbmp);

            //graphics.FillEllipse(new Pen(Color.Red).Brush, new Rectangle(rockcentre.X - 5, rockcentre.Y - 5, 10, 10));

        }
        public static Point PolygonCentre(Point[] bounds)
        {
            float xSum = 0;
            float ySum = 0;
            int numPoints = bounds.Length;

            for (int i = 0; i < numPoints; i++)
            {
                xSum += bounds[i].X;
                ySum += bounds[i].Y;
            }

            float centerX = xSum / numPoints;
            float centerY = ySum / numPoints;

            return new Point((int)centerX, (int)centerY);
        }

        static double DistanceFromPointToPolygon(Point point, Point[] polygon)
        {
            double minDistance = double.MaxValue;

            for (int i = 0; i < polygon.Length; i++)
            {
                Point p1 = polygon[i];
                Point p2 = polygon[(i + 1) % polygon.Length];

                double distance = DistanceFromPointToLineSegment(point, p1, p2);
                minDistance = Math.Min(minDistance, distance);
            }

            return minDistance;
        }

        static double DistanceFromPointToLineSegment(Point p, Point p1, Point p2)
        {
            double A = p.X - p1.X;
            double B = p.Y - p1.Y;
            double C = p2.X - p1.X;
            double D = p2.Y - p1.Y;

            double dot = A * C + B * D;
            double lenSq = C * C + D * D;
            double param = (lenSq != 0) ? dot / lenSq : -1;

            double xx, yy;

            if (param < 0)
            {
                xx = p1.X;
                yy = p1.Y;
            }
            else if (param > 1)
            {
                xx = p2.X;
                yy = p2.Y;
            }
            else
            {
                xx = p1.X + param * C;
                yy = p1.Y + param * D;
            }

            double dx = p.X - xx;
            double dy = p.Y - yy;
            return Math.Sqrt(dx * dx + dy * dy);
        }
    }
}