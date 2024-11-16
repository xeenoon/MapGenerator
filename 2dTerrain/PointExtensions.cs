using System.Drawing;
namespace TerrainGenerator
{
    public static class PointExtensions
    {
        // Add two Points
        public static Point Add(this Point a, Point b) => new Point(a.X + b.X, a.Y + b.Y);

        // Subtract two Points
        public static Point Subtract(this Point a, Point b) => new Point(a.X - b.X, a.Y - b.Y);

        // Dot product of two Points
        public static int DotProduct(this Point a, Point b) => a.X * b.X + a.Y * b.Y;

        // Scale a Point by an integer
        public static Point Scale(this Point p, int scalar) => new Point(p.X * scalar, p.Y * scalar);

        // Scale a Point by a float (result will be cast back to int)
        public static Point Scale(this Point p, float scalar) =>
            new Point((int)(p.X * scalar), (int)(p.Y * scalar));
    }

    public static class PointFExtensions
    {
        // Add two PointFs
        public static PointF Add(this PointF a, PointF b) => new PointF(a.X + b.X, a.Y + b.Y);

        // Subtract two PointFs
        public static PointF Subtract(this PointF a, PointF b) => new PointF(a.X - b.X, a.Y - b.Y);

        // Dot product of two PointFs
        public static float DotProduct(this PointF a, PointF b) => a.X * b.X + a.Y * b.Y;

        // Scale a PointF by a float
        public static PointF Scale(this PointF p, float scalar) => new PointF(p.X * scalar, p.Y * scalar);

        // Scale a PointF by an integer
        public static PointF Scale(this PointF p, int scalar) => new PointF(p.X * scalar, p.Y * scalar);
        public static float Magnitude(this PointF p) => (float)Math.Sqrt(p.X * p.X + p.Y * p.Y);
    }
}