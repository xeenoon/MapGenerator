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
using System.Collections.Immutable;
using System.Drawing.Imaging;

namespace TerrainGenerator
{
    public class Rock : ProceduralShape
    {
        public static Bitmap rock;
        public static BitmapData rockbmp;
        public unsafe static byte* rockbmp_scan0;
        static string exePath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);

        static string filepath = exePath + "\\images\\rock.jpg";// + random.Next(1,7).ToString() + ".jpg";

        public unsafe static void Setup()
        {
            rock = (Bitmap)Image.FromFile(filepath);
            rockbmp = rock.LockBits(new Rectangle(0, 0, rock.Width, rock.Height), ImageLockMode.ReadWrite, PixelFormat.Format32bppRgb);
            rockbmp_scan0 = (byte*)rockbmp.Scan0;

        }
        public Rock()
        {

        }
        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr memcpy(IntPtr dest, IntPtr src, UIntPtr count);

        public unsafe void Draw(BitmapData resultbmp)
        {
            //Find a bounds around the bounds of the rock
            Point topleft = new Point(bounds.OrderBy(p => p.X).FirstOrDefault().X, bounds.OrderBy(p => p.Y).FirstOrDefault().Y);
            Point bottomright = new Point(bounds.OrderByDescending(p => p.X).FirstOrDefault().X, bounds.OrderByDescending(p => p.Y).FirstOrDefault().Y);

            var resultbmp_scan0 = (byte*)resultbmp.Scan0;


            const int shadowdst = 20;
            const int shinedst = 40;
            Point rockcentre = bounds.ToArray().PolygonCentre();
            Filter filter = Filter.RandomFilter();
            for (int x = topleft.X - shadowdst; x < bottomright.X + shadowdst; ++x)
            {
                for (int y = topleft.Y - shadowdst; y < bottomright.Y + shadowdst; ++y)
                {
                    if (x >= resultbmp.Width || y >= resultbmp.Height || x < 0 || y < 0)
                    {
                        continue;
                    }
                    const int BYTES_PER_PIXEL = 4;
                    bool inpolygon = resultbmp_scan0[x * BYTES_PER_PIXEL + y * resultbmp.Stride + 2] == 255;
                    if (inpolygon) // Am I in the polygon?
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
                            double shadowFactor = 1 + ((1.0 / shinestrength) * (1 - (centredst / shinedst)));
                            //double shadowFactor = centredst / (shadowdst * 3) + (1 - (1.0 / 3));

                            resultbmp_scan0[resultIndex] = (byte)Math.Min(b * shadowFactor, 255);
                            resultbmp_scan0[resultIndex + 1] = (byte)Math.Min(g * shadowFactor, 255);
                            resultbmp_scan0[resultIndex + 2] = (byte)Math.Min(r * shadowFactor, 255);
                        }
                        filter.ApplyFilter(resultbmp_scan0 + resultIndex);
                        // Set alpha to 255
                        resultbmp_scan0[resultIndex + 3] = 255;
                    }

                    var distance = DistanceTo(new Point(x, y));
                    if (distance <= shadowdst && distance != -1 && (distance == 0 ? inpolygon : true)) //Darken on edges
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
        }
    }
}