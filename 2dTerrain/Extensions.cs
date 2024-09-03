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
        public static PointF UnitVector(this PointF p)
        {
            var magnitude = Math.Sqrt(p.X * p.X + p.Y * p.Y);
            if (magnitude == 0)
            {
                return new PointF(0, 0); // To handle the case when the point is at the origin
            }
            return new PointF((float)(p.X / magnitude), (float)(p.Y / magnitude));
        }
        public static PointF Perpendicular(this PointF p)
        {
            // Returns a vector perpendicular to the original vector (rotated 90 degrees counterclockwise)
            return new PointF(-p.Y, p.X);
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
        public static PointF[] ToPolygon(this RectangleF rect, float bezel = 0, float bump = 0)
        {
            const int detaillevel = 10;
            List<PointF> result = new List<PointF>();
            //Start on the left hand side, drawing points from rect.left
            for (int i = 0; i < detaillevel + 1; ++i) //+1 to add a point at the bottom left
            {
                //Assume we start at the minimum of the sin curve, doing on full sin rotation
                //y = sin(x-pi/2)
                //x is min at 0, 2pi
                double x = 2 * i * Math.PI / detaillevel;

                var curve = (float)Math.Sin(x - Math.PI / 2) * bump / 2;

                float y = (i * (rect.Bottom - rect.Top) / (detaillevel)) + rect.Top;
                result.Add(new PointF(rect.Left + curve, y));
            }

            //Add a point at the bottom right
            result.Add(new PointF(rect.Right, rect.Bottom));

            //Start on the right hand side, drawing points from rect.right
            for (int i = 0; i < detaillevel + 1; ++i) //+1 to add a point to the top right
            {
                //Assume we start at the minimum of the sin curve, doing on full sin rotation
                //y = sin(x)
                //x is 0 at 0, pi
                double x = i * Math.PI / detaillevel;

                var curve = (float)Math.Sin(x) * bezel;
                float y = rect.Bottom - (i * (rect.Bottom - rect.Top) / detaillevel); //draw bottom up
                result.Add(new PointF(rect.Right + curve, y));
            }
            return result.ToArray();

            // Calculate the basic four corners of the rectangle
            PointF topLeft = new PointF(rect.Left, rect.Top);
            PointF topRight = new PointF(rect.Right, rect.Top);
            PointF bottomRight = new PointF(rect.Right, rect.Bottom);
            PointF bottomLeft = new PointF(rect.Left, rect.Bottom);

            // Apply the bezel (inward curve on the right side)
            if (bezel > 0)
            {
                topRight.X -= bezel;
                bottomRight.X -= bezel;
            }

            // Apply the bump (outward curve on the left side)
            if (bump > 0)
            {
                topLeft.X -= bump;
                bottomLeft.X -= bump;
            }

            // Return the points as a polygon
            return new PointF[]
            {
                topLeft,
                topRight,
                bottomRight,
                bottomLeft
            };
        }
        public static PointF[] Rotate(this PointF[] polygon, float theta, PointF? centre = null)
        {
            float centroidX = 0;
            float centroidY = 0;
            if (centre.HasValue)
            {
                centroidX = centre.Value.X;
                centroidY = centre.Value.Y;
            }
            else
            {
                // Calculate the centroid of the polygon
                centroidX = polygon.Average(p => p.X);
                centroidY = polygon.Average(p => p.Y);
            }
            PointF centroid = new PointF(centroidX, centroidY);

            float cosTheta = (float)Math.Cos(theta);
            float sinTheta = (float)Math.Sin(theta);

            PointF[] rotatedPolygon = new PointF[polygon.Length];

            for (int i = 0; i < polygon.Length; i++)
            {
                float x = polygon[i].X - centroid.X;
                float y = polygon[i].Y - centroid.Y;

                // Rotate around the centroid
                float newX = x * cosTheta - y * sinTheta + centroid.X;
                float newY = x * sinTheta + y * cosTheta + centroid.Y;

                rotatedPolygon[i] = new PointF(newX, newY);
            }

            return rotatedPolygon;
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
        public static float Angle(this PointF vector)
        {
            return (float)Math.Atan2(vector.Y, vector.X); // Returns angle in radians
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
        public static Point Add(this Point p1, Point p2)
        {
            return new Point(p1.X + p2.X, p1.Y + p2.Y);
        }
    }
}