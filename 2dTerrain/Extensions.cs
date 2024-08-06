namespace TerrainGenerator
{
    public static class Extensions
    {
        public static int DistanceTo(this Point point1, Point point2)
        {
            int xdist = point1.X - point2.X;
            int ydist = point1.Y - point2.Y;
            return (int)Math.Sqrt(xdist*xdist + ydist*ydist);
        }
    }
}