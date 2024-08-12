using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace TerrainGenerator
{
    public unsafe class Polygon : IDisposable
    {
        public Point[] points;
        private int* indices;
        private int len_indices;


        [DllImport("polygon.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern byte* drawPolygonWithTriangles(int* x, int* y, int* indices, int numTriangles, byte* color, byte* bitmap, int width, int height);
        public Polygon(Point[] points)
        {
            if (points.Count() <= 2)
            {
                throw new Exception("Polygon must have at least 3 points");
            }
            this.points = points;

            len_indices = 3 * (points.Length - 2);
            indices = (int*)Marshal.AllocHGlobal(len_indices * sizeof(int));

            for (int i = 2; i < points.Length; ++i)
            {
                int idx = (i - 2) * 3;
                //Imagine a line between p[0] and p[i]
                //Check if it intersects with any other lines

                //Add the triangle to the list
                indices[idx] = 0;
                indices[idx + 1] = i - 1;
                indices[idx + 2] = i;
            }
        }
        public Polygon(Point[] points, int* indices)
        {
            this.points = points;
            len_indices = 3 * (points.Length - 2);

            this.indices = (int*)Marshal.AllocHGlobal(len_indices * sizeof(int));

            Buffer.MemoryCopy(indices, this.indices, len_indices * sizeof(int), len_indices * sizeof(int));
        }
        public Polygon Scale(int n, Point staticoffset)
        {
            return new Polygon(points.ScalePolygon(n, staticoffset), indices);
        }
        public void Draw(byte* color, BitmapData bmp)
        {
            int* xs = (int*)Marshal.AllocHGlobal(points.Length * sizeof(int));
            int* ys = (int*)Marshal.AllocHGlobal(points.Length * sizeof(int));

            for (int i = 0; i < points.Length; ++i)
            {
                xs[i] = points[i].X;
                ys[i] = points[i].Y;
            }
            byte* result = drawPolygonWithTriangles(xs, ys, indices, points.Length - 2, color, (byte*)bmp.Scan0, bmp.Width, bmp.Height);
            Buffer.MemoryCopy(result, (byte*)bmp.Scan0, bmp.Stride * bmp.Height, bmp.Stride * bmp.Height);
            Marshal.FreeHGlobal((IntPtr)xs);
            Marshal.FreeHGlobal((IntPtr)ys);
        }
        public void Draw(Color color, BitmapData bmp)
        {
            byte* bytecolor = (byte*)Marshal.AllocHGlobal(4);
            bytecolor[0] = color.B;
            bytecolor[1] = color.G;
            bytecolor[2] = color.R;
            bytecolor[3] = color.A;
            Draw(bytecolor, bmp);
            Marshal.FreeHGlobal((IntPtr)bytecolor);
        }
        public void DebugDraw(Graphics g)
        {
            Random r = new Random();
            for (int i = 0; i < (len_indices / 3); ++i)
            {
                Point[] todraw = [points[indices[i * 3]], points[indices[i * 3 + 1]], points[indices[i * 3 + 2]]];
                Color randomcolor = Color.FromArgb(255, r.Next(0, 255), r.Next(0, 255), r.Next(0, 255));
                g.FillPolygon(new Pen(randomcolor).Brush, todraw);
            }
        }
        public void Move(Point p)
        {
            points = points.Offset(p);
        }
        public void Dispose()
        {
            Marshal.FreeHGlobal((IntPtr)indices);
        }
    }
}