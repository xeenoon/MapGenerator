using System.Management;

namespace TerrainGenerator
{
    public static class Extensions
    {
        public static Stack<int> Shuffle(this Stack<int> ints) //Shuffles deck
        {
            Random r = new Random();
            Dictionary<int, int> intindices = new Dictionary<int, int>();
            foreach (var i in ints)
            {
                int rand = 0;
                do
                {
                    rand = r.Next();
                } while (intindices.ContainsKey(rand)); //Ensure no duplicate keys despite the fact that the chance is literally 1/4 billion
                intindices.Add(rand, i);
            }
            return new Stack<int>(intindices.OrderBy(c => c.Key).Select(c => c.Value).ToList());
        }
        public static bool HasNvidiaGpu()
        {
            return false;
            try
            {
                // Create a ManagementObjectSearcher object to query for GPU devices
                ManagementObjectSearcher searcher = new ManagementObjectSearcher("SELECT * FROM Win32_VideoController");

                foreach (ManagementObject obj in searcher.Get())
                {
                    // Check the "Name" property for "NVIDIA"
                    string name = obj["Name"].ToString();
                    if (name.Contains("NVIDIA", StringComparison.OrdinalIgnoreCase))
                    {
                        return true;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error querying for GPU information: " + ex.Message);
            }

            return false;
        }
        public static int DistanceTo(this Point point1, Point point2)
        {
            int xdist = point1.X - point2.X;
            int ydist = point1.Y - point2.Y;
            return (int)Math.Sqrt(xdist * xdist + ydist * ydist);
        }
        public static double DistanceTo(this PointF point1, PointF point2)
        {
            double xdist = point1.X - point2.X;
            double ydist = point1.Y - point2.Y;
            return Math.Sqrt(xdist * xdist + ydist * ydist);
        }
        public static Point[] ScalePolygon(this Point[] points, int scaleAmount, Point staticoffset)
        {
            // Calculate the centroid of the polygon
            float centroidX = 0;
            float centroidY = 0;
            foreach (var point in points)
            {
                centroidX += point.X;
                centroidY += point.Y;
            }
            centroidX /= points.Length;
            centroidY /= points.Length;

            PointF centroid = new PointF(centroidX, centroidY);

            // Scale each point towards the centroid
            Point[] scaledPoints = new Point[points.Length];
            for (int i = 0; i < points.Length; i++)
            {
                // Calculate the vector from the centroid to the point
                float vectorX = points[i].X - centroid.X;
                float vectorY = points[i].Y - centroid.Y;

                // Calculate the distance from the centroid to the point
                float distance = (float)Math.Sqrt(vectorX * vectorX + vectorY * vectorY);

                // Calculate the new distance after scaling
                float newDistance = Math.Max(0, distance + scaleAmount);

                // Calculate the scale factor
                float scale = newDistance / distance;

                // Scale the point
                scaledPoints[i] = new Point(
                    (int)(centroid.X + vectorX * scale) + staticoffset.X,
                    (int)(centroid.Y + vectorY * scale) + staticoffset.Y
                );
            }

            return scaledPoints;
        }
        public static Point[] Offset(this Point[] points, Point offset)
        {
            Point[] result = new Point[points.Length];
            for (int i = 0; i < points.Length; ++i)
            {
                result[i] = new Point(points[i].X + offset.X, points[i].Y + offset.Y);
            }
            return result;
        }
        public static Point PolygonCentre(this Point[] bounds)
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

        public static double DistanceFromPointToPolygon(this Point[] polygon, Point point)
        {
            double minDistance = double.MaxValue;

            for (int i = 0; i < polygon.Length; i++)
            {
                Point p1 = polygon[i];
                Point p2 = polygon[(i + 1) % polygon.Length];

                double distance = point.DistanceFromPointToLineSegment(p1, p2);
                minDistance = Math.Min(minDistance, distance);
            }

            return minDistance;
        }

        static double DistanceFromPointToLineSegment(this Point p, Point p1, Point p2)
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

        public unsafe static void BlendColors(byte* from, byte* to, double blendfactor)
        {

            //Start from the color 0 blend towards color 1
            int r_difference = to[2] - from[2];
            from[2] += (byte)(r_difference * blendfactor); //Blend factor of 0 will be original color, blend factor of 1 will e completely the next color
            int g_difference = to[1] - from[1];
            from[1] += (byte)(g_difference * blendfactor); //Blend factor of 0 will be original color, blend factor of 1 will e completely the next color
            int b_difference = to[0] - from[0];
            from[0] += (byte)(b_difference * blendfactor); //Blend factor of 0 will be original color, blend factor of 1 will e completely the next color
        }

        public static bool PointInPolygon(this Point[] polygon, Point point) //Genuinely never use this method, use Graphics.DrawPolygon to bake the pixel values
        {
            int n = polygon.Length;
            bool inside = false;

            for (int i = 0, j = n - 1; i < n; j = i++)
            {
                if (((polygon[i].Y > point.Y) != (polygon[j].Y > point.Y)) &&
                    (point.X < (polygon[j].X - polygon[i].X) * (point.Y - polygon[i].Y) / (double)(polygon[j].Y - polygon[i].Y) + polygon[i].X))
                {
                    inside = !inside;
                }
            }

            return inside;
        }

        public static bool Intersects(this Point[] polygon1, Point[] polygon2) //This is genuinely the most inefficient function i have seen in my entire life
        {
            // Check if any vertex of polygon1 is inside polygon2
            foreach (var p in polygon1)
            {
                if (polygon2.PointInPolygon(p))
                {
                    return true;
                }
            }

            // Check if any vertex of polygon2 is inside polygon1
            foreach (var q in polygon2)
            {
                if (polygon1.PointInPolygon(q))
                {
                    return true;
                }
            }

            return false;
        }
    }
}