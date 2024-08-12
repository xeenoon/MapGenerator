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
            //Point topleft = new Point(bounds.OrderBy(p => p.X).FirstOrDefault().X, bounds.OrderBy(p => p.Y).FirstOrDefault().Y);
            //Point bottomright = new Point(bounds.OrderByDescending(p => p.X).FirstOrDefault().X, bounds.OrderByDescending(p => p.Y).FirstOrDefault().Y);

            Point topleft = new Point(rect_bounds.Left, rect_bounds.Top);
            Point bottomright = new Point(rect_bounds.Right, rect_bounds.Bottom);

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
                    bool inpolygon = Contains(new Point(x, y));
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
        [DllImport("vectorexample.dll", CallingConvention = CallingConvention.Cdecl)]
        public static unsafe extern byte* CudaDraw(
            int* rockcentreXs, int* rockcentreYs, int* topleftXs, int* topleftYs, int* bottomrightXs, int* bottomrightYs,
            int* bakedrectangleLefts, int* bakedrectangleTops, int* bakedrectangleWidths, int* bakedrectangleHeights,
            IntPtr* bakeddistances_dataScan0s, IntPtr* bakedbounds_dataScan0s,
            IntPtr* filters, IntPtr resultbmp_scan0, int resultwidth, int resultheight, IntPtr rockbmp_scan0, int rockWidth, int rockHeight, int numItems, int maxrockwidth, int maxrockheight);

        [DllImport("msvcrt.dll", EntryPoint = "memcpy", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr memcpy(IntPtr dest, IntPtr src, int count);

        public static unsafe void CudaDraw(Rock[,] rockgrid, int width, int height, BitmapData resultbmp)
        {
            if (Extensions.HasNvidiaGpu())
            {
                int groupsize = 2;
                for (int g = 0; g < groupsize; ++g)
                {
                    int numItems = (width * height) / groupsize;

                    // Allocate memory for the pointers
                    int* centreXs = (int*)Marshal.AllocHGlobal(numItems * sizeof(int));
                    int* centreYs = (int*)Marshal.AllocHGlobal(numItems * sizeof(int));

                    int* topleftXs = (int*)Marshal.AllocHGlobal(numItems * sizeof(int));
                    int* topleftYs = (int*)Marshal.AllocHGlobal(numItems * sizeof(int));

                    int* bottomrightXs = (int*)Marshal.AllocHGlobal(numItems * sizeof(int));
                    int* bottomrightYs = (int*)Marshal.AllocHGlobal(numItems * sizeof(int));

                    int* bakedrectangleLefts = (int*)Marshal.AllocHGlobal(numItems * sizeof(int));
                    int* bakedrectangleTops = (int*)Marshal.AllocHGlobal(numItems * sizeof(int));
                    int* bakedrectangleWidths = (int*)Marshal.AllocHGlobal(numItems * sizeof(int));
                    int* bakedrectangleHeights = (int*)Marshal.AllocHGlobal(numItems * sizeof(int));

                    // Allocate memory for the byte pointers
                    byte** bakeddistances_dataScan0s = (byte**)Marshal.AllocHGlobal(numItems * sizeof(byte*));
                    byte** bakedbounds_dataScan0s = (byte**)Marshal.AllocHGlobal(numItems * sizeof(byte*));

                    int** filters = (int**)Marshal.AllocHGlobal(numItems * sizeof(int*));

                    // Populate the allocated memory
                    int idx = 0;
                    for (int i = g; i < numItems * groupsize; i += groupsize)
                    {
                        int x = i % width;
                        int y = i / width;

                        var filter = Filter.RandomFilter();
                        filters[idx] = filter.GetArry();
                        // Copy filter array data to the GPU; not shown here, assumed to be done elsewhere

                        centreXs[idx] = rockgrid[x, y].centre.X;
                        centreYs[idx] = rockgrid[x, y].centre.Y;

                        topleftXs[idx] = rockgrid[x, y].bakedrectangle.Left;
                        topleftYs[idx] = rockgrid[x, y].bakedrectangle.Top;

                        bottomrightXs[idx] = rockgrid[x, y].bakedrectangle.Right;
                        bottomrightYs[idx] = rockgrid[x, y].bakedrectangle.Bottom;

                        bakedrectangleLefts[idx] = rockgrid[x, y].bakedrectangle.Left;
                        bakedrectangleTops[idx] = rockgrid[x, y].bakedrectangle.Top;

                        bakedrectangleWidths[idx] = rockgrid[x, y].bakedrectangle.Width;
                        bakedrectangleHeights[idx] = rockgrid[x, y].bakedrectangle.Height;

                        bakeddistances_dataScan0s[idx] = (byte*)rockgrid[x, y].bakeddistances_data.Scan0;
                        bakedbounds_dataScan0s[idx] = (byte*)rockgrid[x, y].bakedbounds_data.Scan0;
                        ++idx;
                    }

                    int maxrockwidth = rockgrid.Cast<Rock>().Max(r => r.bakedrectangle.Width);
                    int maxrockheight = rockgrid.Cast<Rock>().Max(r => r.bakedrectangle.Height);
                    var result = CudaDraw(centreXs, centreYs, topleftXs, topleftYs, bottomrightXs, bottomrightYs,
                        bakedrectangleLefts, bakedrectangleTops, bakedrectangleWidths, bakedrectangleHeights,
                        (IntPtr*)bakeddistances_dataScan0s, (IntPtr*)bakedbounds_dataScan0s, (IntPtr*)filters, resultbmp.Scan0, resultbmp.Width, resultbmp.Height,
                        rockbmp.Scan0, rockbmp.Width, rockbmp.Height, numItems, maxrockwidth, maxrockheight);

                    memcpy(resultbmp.Scan0, (nint)result, resultbmp.Stride * resultbmp.Height);


                    // Clean up allocated memory
                    Marshal.FreeHGlobal((IntPtr)centreXs);
                    Marshal.FreeHGlobal((IntPtr)centreYs);
                    Marshal.FreeHGlobal((IntPtr)topleftXs);
                    Marshal.FreeHGlobal((IntPtr)topleftYs);
                    Marshal.FreeHGlobal((IntPtr)bottomrightXs);
                    Marshal.FreeHGlobal((IntPtr)bottomrightYs);
                    Marshal.FreeHGlobal((IntPtr)bakedrectangleLefts);
                    Marshal.FreeHGlobal((IntPtr)bakedrectangleTops);
                    Marshal.FreeHGlobal((IntPtr)bakedrectangleWidths);
                    Marshal.FreeHGlobal((IntPtr)bakedrectangleHeights);
                    Marshal.FreeHGlobal((IntPtr)bakeddistances_dataScan0s);
                    Marshal.FreeHGlobal((IntPtr)bakedbounds_dataScan0s);
                }
            }
            else
            {
                foreach (var rock in rockgrid)
                {
                    rock.Draw(resultbmp);
                }
            }
        }
    }
}