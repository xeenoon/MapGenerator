using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Security.Cryptography.X509Certificates;
using Microsoft.VisualBasic;

namespace TerrainGenerator
{
    partial class Caterpillar
    {
        public List<PointF> spine = new List<PointF>();
        public double[] olddistances;
        public static int pointsperbump = 8;
        public static int sectionwidth = 20;
        public static int sectionheight = 50;
        private List<Leg> legs = new List<Leg>();
        public Tail tail;
        public Mouth mouth;
        public Caterpillar(int length, Point head)
        {
            if (length <= 2)
            {
                throw new Exception("Length must be over 3");
            }
            //Calculate the old distances
            olddistances = new double[length]; //array is one longer than required to avoid weird math later on. 
            //'i' index for distnace is the same as the tail of the dragged point

            for (int i = 0; i < length; ++i)
            {
                spine.Add(new PointF(head.X - sectionwidth * i, head.Y));
                if (i >= 1)
                {
                    olddistances[i] = spine[i].DistanceTo(spine[i - 1]);
                }
            }
            tail = new Tail(10, spine.Last(), sectionwidth, sectionheight);
            mouth = new Mouth(head, (int)(sectionwidth * 3f));
            legs = Leg.BuildLegs(spine.ToArray(), sectionwidth * 5);
        }
        int time = 0;
        public void MoveTowards(Point p)
        {
            ++time;
            //Assume it will be called once a frame
            const double speed = 10;

            //Rotate the head towards the mouse
            const double rotationspeed = 0.05;
            spine[0] = RotateTowards(spine[1], spine[0], p, rotationspeed);
            double sinmodifier = 0;
            if (((PointF)p).DistanceTo(spine[0]) > 100)
            {
                //sinmodifier = Math.Sin(time/10.0) / 4;
            }

            double neckangle = CalculateAngle(spine[1], spine[0]) + sinmodifier;
            var newhead = DragPoint(spine[0], neckangle, speed);

            mouth.head = newhead;
            mouth.start_jawtop = new PointF((float)(newhead.X + Math.Cos(Mouth.jawangle + neckangle) * mouth.jawlength), (float)(newhead.Y + Math.Sin(Mouth.jawangle + neckangle) * mouth.jawlength));
            mouth.start_jawbot = new PointF((float)(newhead.X + Math.Cos(-Mouth.jawangle + neckangle) * mouth.jawlength), (float)(newhead.Y + Math.Sin(-Mouth.jawangle + neckangle) * mouth.jawlength));
            mouth.jawmid = new PointF((float)(newhead.X + Math.Cos(neckangle) * mouth.jawlength), (float)(newhead.Y + Math.Sin(neckangle) * mouth.jawlength));

            spine[0] = newhead;
            mouth.Bite();

            //Recursively go through all spine points, moving it towards the LAST one
            for (int i = 1; i < spine.Count(); ++i)
            {
                //Drag myself towards this new angle by speed
                PointF dragged = DragPoint(spine[i], spine[i - 1], speed, olddistances[i]);
                var leg = legs.Where(l => l.spineconnection.DistanceTo(spine[i]) < 1).ToList();
                if (leg.Count() != 0)
                {
                    for (int j = 0; j < leg.Count; ++j)
                    {
                        var side = leg[j];

                        double angle_front = CalculateAngle(spine[i], spine[i - 1]);
                        double angle_behind = CalculateAngle(spine[i + 1], spine[i]);
                        double averageangle = angle_front;

                        double perpindicular = averageangle + (j == 0 ? Math.PI / 2 : -Math.PI / 2); //If j is zero its the "left" if its one its on the "right"

                        side.startfoot = new PointF((float)(dragged.X + Math.Cos(perpindicular) * side.length), (float)(dragged.Y + Math.Sin(perpindicular) * side.length));

                        var legmovedst = Math.Min(speed, spine[i].DistanceTo(spine[i - 1]) - olddistances[i]); //Throttle speed

                        side.WalkCycle(dragged, (float)angle_front, legmovedst);
                    }
                }
                if (i == spine.Count() - 1)
                {
                    for (int j = 0; j < tail.points.Count(); ++j)
                    {
                        double angle_front = CalculateAngle(spine[i], spine[i - 1]);
                        tail.start_points[j] = new PointF((float)(dragged.X - Math.Cos(angle_front) * tail.sectionwidth * j),
                                                          (float)(dragged.Y - Math.Sin(angle_front) * tail.sectionwidth * j));
                    }
                    tail.Swing();
                }

                spine[i] = dragged;
            }
        }
        private static PointF DragPoint(PointF p0, PointF p1, double speed, double minDistance)
        {
            speed = Math.Min(speed, p0.DistanceTo(p1) - minDistance); //Throttle speed
            if (speed < 0)
            {
                return p0;
            }
            double angle = CalculateAngle(p0, p1);
            PointF dp = new PointF((float)(Math.Cos(angle) * speed), (float)(Math.Sin(angle) * speed));
            PointF result = new PointF(p0.X + dp.X, p0.Y + dp.Y);


            return result;
        }
        private static PointF DragPoint(PointF p, double angle, double speed)
        {
            PointF dp = new PointF((float)(Math.Cos(angle) * speed), (float)(Math.Sin(angle) * speed));
            return new PointF(p.X + dp.X, p.Y + dp.Y);
        }
        public static PointF RotateTowards(PointF p0, PointF p1, PointF p2, double rotationspeed)
        {
            double headangle = CalculateAngle(p0, p1);
            double mouseangle = CalculateAngle(p0, p2);

            double newangle;
            if (Math.Abs(mouseangle - headangle) < rotationspeed)
            {
                newangle = mouseangle;
            }
            else
            {
                double difference = mouseangle - headangle;

                if (difference > Math.PI) difference -= 2 * Math.PI;
                else if (difference < -Math.PI) difference += 2 * Math.PI;

                newangle = headangle + (difference > 0 ? rotationspeed : -rotationspeed);
            }
            return new PointF((float)(Math.Cos(newangle) * p0.DistanceTo(p1)) + p0.X, (float)(Math.Sin(newangle) * p0.DistanceTo(p1) + p0.Y));
        }
        public static double CalculateAngle(PointF p1, PointF p2)
        {
            double deltaY = p2.Y - p1.Y;
            double deltaX = p2.X - p1.X;
            double angle = Math.Atan2(deltaY, deltaX);

            // Adjust angle to be between 0 and 360 degrees
            if (angle < 0)
            {
                angle += Math.PI * 2;
            }
            return angle;
        }
        public PointF[] CurvePolygon(PointF centre, int centrewidth, int length, int endoffset, double angle, int detail = 5)
        {
            List<PointF> result = new List<PointF>();
            result.Add(new PointF(centre.X, centre.Y));
            double perpindicular = angle + Math.PI / 2;
            float offsetX = (float)(endoffset * Math.Cos(perpindicular));
            float offsetY = (float)(endoffset * Math.Sin(perpindicular));

            //Left line
            for (int i = 0; i < detail; ++i)
            {
                double percentagedistance = i / (double)detail;
                double sinoffset = Math.Sin(Math.PI / 2 * percentagedistance);

                // Calculate the new x and y positions using the angle
                float x = (float)(centre.X - Math.Abs((length / 2) * percentagedistance * Math.Cos(angle) + offsetX * sinoffset));
                float y = (float)(centre.Y - Math.Abs((length / 2) * percentagedistance * Math.Sin(angle) + offsetY * sinoffset));
                result.Add(new PointF(x, y));
            }
            for (int i = 1; i < detail + 1; ++i)
            {
                // Reflects a vector `p` about the line at `angle` degrees
                var p = result[i];
                var centre_p = new PointF(p.X - centre.X, p.Y - centre.Y);

                // Convert angle to radians
                double angleRadians = angle * Math.PI / 180.0;

                // Calculate the reflection matrix components
                double cosAngle = Math.Cos(angleRadians);
                double sinAngle = Math.Sin(angleRadians);

                // Compute the reflected vector
                double x = centre_p.X;
                double y = centre_p.Y;
                double reflectedX = x * (cosAngle * cosAngle - sinAngle * sinAngle) + y * 2 * cosAngle * sinAngle;
                double reflectedY = x * 2 * cosAngle * sinAngle + y * (sinAngle * sinAngle - cosAngle * cosAngle);

                // Create the reflected point
                var reflectedPoint = new PointF((float)(centre.X + reflectedX), (float)(centre.Y + reflectedY));

                result.Add(reflectedPoint);
            }

            return result
                   .OrderBy(p => Math.Atan2(p.Y - centre.Y, p.X - centre.X)) // Calculate angle of each point with respect to the centre
                   .ToArray();
        }

        public void Draw(Bitmap result)
        {
            Graphics g = Graphics.FromImage(result);
            PointF[] points = new RectangleF(spine[0].X - sectionwidth, spine[0].Y - sectionheight / 8, sectionwidth * 2, sectionheight / 4).ToPolygon(10, 10);
            var angle = (float)CalculateAngle(spine[1], spine[0]);
            points = points.Rotate(angle);

            //g.DrawPolygon(new Pen(Color.Black), points);
            for (int i = 1; i < spine.Count(); ++i)
            {
                if (i % 5 != 0)
                {
                  //  continue;
                }
                int width = sectionwidth;
                int height = sectionheight;
                const double tailsize = 0.2;
                if (((double)i) / spine.Count() >= (1 - tailsize)) //Only stretch down the last 0.1
                {
                    double scalar = (spine.Count() - i) / (tailsize * spine.Count());
                    width = (int)(width * scalar);
                    height = (int)(height * scalar);
                }

                var p = spine[i];
                float spinesize = 0.02f;
                angle = (float)CalculateAngle(spine[i], spine[i - 1]);
                points = CurvePolygon(p, sectionwidth, sectionheight, 20, angle + Math.PI / 2);
                //points = new RectangleF(p.X - (width / 2) * spinesize, p.Y - height / 2, width * spinesize, height).ToPolygon(15, 20);
                points = points.Rotate(angle, spine[i]);

                for (int j = 0; j < points.Length - 1; ++j)
                {
                    g.DrawLine(new Pen(Color.Red, 3), points[j], points[j + 1]);
                }

                //g.FillPolygon(new Pen(Color.Black).Brush, points);
                g.DrawLine(new Pen(Color.Black, spinesize * width / 2), spine[i], spine[i - 1]);

                //g.FillEllipse(new Pen(Color.Red).Brush, new RectangleF(p.X - dotsize / 2, p.Y - dotsize / 2, dotsize, dotsize));
            }
            foreach (var leg in legs)
            {
                g.DrawLine(new Pen(Color.Black, 4), leg.spineconnection, leg.knee);
                g.DrawLine(new Pen(Color.Black, 4), leg.knee, leg.foot);
                foreach (var toe in leg.toes)
                {
                    g.DrawLine(new Pen(Color.Black, 4), leg.foot, toe);
                }

                //g.FillEllipse(new Pen(Color.Red).Brush, new RectangleF(leg.knee.X - 5, leg.knee.Y - 5, 10, 10));
                //      g.FillEllipse(new Pen(Color.Green).Brush, new RectangleF(leg.foot.X - 5, leg.foot.Y - 5, 10, 10));
                //      g.FillEllipse(new Pen(Color.Red).Brush, new RectangleF(leg.startfoot.X - 5, leg.startfoot.Y - 5, 10, 10));
                //      g.FillEllipse(new Pen(Color.Blue).Brush, new RectangleF(leg.swingdestination.X - 5, leg.swingdestination.Y - 5, 10, 10));
            }
            for (int i = 1; i < tail.length; ++i)
            {
                g.DrawLine(new Pen(Color.Black, 4), tail.points[i - 1], tail.points[i]);
            }
            var lastpoint = tail.points[tail.points.Count() - 1];
            float radius = 10;

            g.FillEllipse(new Pen(Color.Black).Brush, new RectangleF(lastpoint.X - radius, lastpoint.Y - radius, radius * 2, radius * 2));

            // Calculate and draw the 6 triangles
            int numTriangles = 6;
            for (int i = 0; i < numTriangles; i++)
            {
                // Calculate the angle for this triangle
                double triangle_angle = i * 2 * Math.PI / numTriangles;

                // Calculate the two points on the circle where the triangle base meets
                PointF p1 = new PointF(
                    lastpoint.X + radius * (float)Math.Cos(triangle_angle - Math.PI / numTriangles),
                    lastpoint.Y + radius * (float)Math.Sin(triangle_angle - Math.PI / numTriangles));

                PointF p2 = new PointF(
                    lastpoint.X + radius * (float)Math.Cos(triangle_angle + Math.PI / numTriangles),
                    lastpoint.Y + radius * (float)Math.Sin(triangle_angle + Math.PI / numTriangles));

                // Calculate the point where the triangle's apex is (outside the circle)
                PointF apex = new PointF(
                    lastpoint.X + 1.5f * radius * (float)Math.Cos(triangle_angle),
                    lastpoint.Y + 1.5f * radius * (float)Math.Sin(triangle_angle));

                // Draw the triangle
                PointF[] trianglePoints = { p1, p2, apex };
                g.FillPolygon(new Pen(Color.Black).Brush, trianglePoints);
            }

            var topmid = CalculatePerpendicularMidpoint(mouth.head, mouth.jawtop, 5, false);
            var top_quarter = CalculatePerpendicularMidpoint(mouth.head, topmid, 8, false);
            var top_third_quarter = CalculatePerpendicularMidpoint(topmid, mouth.jawtop, 8, false);

            var topmid_outside = CalculatePerpendicularMidpoint(mouth.head, mouth.jawtop, 20, false);
            var top_quarter_outside = CalculatePerpendicularMidpoint(mouth.head, topmid, 16, false);
            var top_third_quarter_outside = CalculatePerpendicularMidpoint(topmid, mouth.jawtop, 16, false);

            var botmid = CalculatePerpendicularMidpoint(mouth.head, mouth.jawbot, 5, true);
            var bot_quarter = CalculatePerpendicularMidpoint(mouth.head, botmid, 8, true);
            var bot_third_quarter = CalculatePerpendicularMidpoint(botmid, mouth.jawbot, 8, true);

            var botmid_outside = CalculatePerpendicularMidpoint(mouth.head, mouth.jawbot, 20, true);
            var bot_quarter_outside = CalculatePerpendicularMidpoint(mouth.head, botmid, 16, true);
            var bot_third_quarter_outside = CalculatePerpendicularMidpoint(botmid, mouth.jawbot, 16, true);

            PointF[] topjaw = new PointF[] { mouth.head, top_quarter, topmid, top_third_quarter, mouth.jawtop, top_third_quarter_outside, topmid_outside, top_quarter_outside };
            g.FillPolygon(new Pen(Color.Black).Brush, topjaw);

            PointF[] botjaw = new PointF[] { mouth.head, bot_quarter, botmid, bot_third_quarter, mouth.jawbot, bot_third_quarter_outside, botmid_outside, bot_quarter_outside };
            g.FillPolygon(new Pen(Color.Black).Brush, botjaw);

            PointF[] jawtop = new PointF[] { mouth.head, topmid, mouth.jawtop };
            PointF[] jawbot = new PointF[] { mouth.head, botmid, mouth.jawtop };

        }
        public static PointF CalculatePerpendicularMidpoint(PointF point1, PointF point2, float offset, bool right)
        {
            // Calculate the midpoint between point1 and point2
            var midpoint = new PointF(
                (point1.X + point2.X) / 2,
                (point1.Y + point2.Y) / 2
            );

            // Calculate the vector from point1 to point2
            var vectorX = point2.X - point1.X;
            var vectorY = point2.Y - point1.Y;

            // Calculate the perpendicular vector
            var perpX = right ? vectorY : -vectorY;
            var perpY = right ? -vectorX : vectorX;

            // Normalize the perpendicular vector
            var length = Math.Sqrt(perpX * perpX + perpY * perpY);
            perpX /= (float)length;
            perpY /= (float)length;

            // Move the midpoint along the perpendicular vector
            midpoint.X += offset * perpX;
            midpoint.Y += offset * perpY;

            return midpoint;
        }
    }
}