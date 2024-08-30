using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;

namespace TerrainGenerator
{
    class Caterpillar
    {
        public List<PointF> spine = new List<PointF>();
        public double[] olddistances;
        public static int pointsperbump = 8;
        public static int sectionwidth = 20;
        public static int sectionheight = 50;
        private List<Leg> legs = new List<Leg>();
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
            legs = Leg.BuildLegs(spine.ToArray(), sectionwidth * 5);
        }
        public void MoveTowards(Point p)
        {
            //Assume it will be called once a frame
            const double speed = 10;

            //Rotate the head towards the mouse
            const double rotationspeed = 0.05;
            spine[0] = RotateTowards(spine[1], spine[0], p, rotationspeed);
            spine[0] = DragPoint(spine[0], CalculateAngle(spine[1], spine[0]), speed);

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
        public void Draw(Bitmap result)
        {
            Graphics g = Graphics.FromImage(result);
            PointF[] points = new RectangleF(spine[0].X - sectionwidth, spine[0].Y - sectionheight / 8, sectionwidth * 2, sectionheight / 4).ToPolygon(10, 10);
            var angle = (float)CalculateAngle(spine[1], spine[0]);
            points = points.Rotate(angle);

            g.DrawPolygon(new Pen(Color.Black), points);
            for (int i = 1; i < spine.Count(); ++i)
            {
                var p = spine[i];
                int dotsize = 10;
                float spinesize = 0.02f;
                points = new RectangleF(p.X - sectionwidth / 2 * spinesize, p.Y - sectionheight / 2, sectionwidth * spinesize, sectionheight).ToPolygon(20, 20);
                angle = (float)CalculateAngle(spine[i], spine[i - 1]);
                points = points.Rotate(angle);
                g.FillPolygon(new Pen(Color.Black).Brush, points);
                g.DrawLine(new Pen(Color.Black, spinesize * sectionwidth / 2), spine[i], spine[i - 1]);

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
        }

        class Leg
        {
            public PointF spineconnection;
            public PointF knee;
            public PointF foot;
            public PointF startfoot;
            public PointF[] toes = new PointF[3];
            public PointF[] toe_start = new PointF[3];
            public int length;
            public int toelength;
            private Leg(PointF spineconnection, PointF kee, PointF foot, PointF[] toes, int length)
            {
                this.spineconnection = spineconnection;
                this.knee = kee;
                this.foot = foot;
                this.length = length;
                toelength = length / 6;
                this.toes = toes;

                startfoot = foot;
            }
            public static List<Leg> BuildLegs(PointF[] spine, int length)
            {
                List<Leg> legs = new List<Leg>();
                const int legdst = 5;
                for (int i = 1; i < spine.Length / legdst; ++i) //Start at one to not draw a leg on the head
                {
                    PointF spineconnection = spine[i * legdst];

                    //Assume spine[] y's are all the same
                    for (int direction = -1; direction < 2; direction += 2)
                    {
                        PointF knee = new PointF(spineconnection.X, spineconnection.Y + direction * length / 2);
                        PointF foot = new PointF(spineconnection.X, spineconnection.Y + direction * length);
                        PointF[] toes = new PointF[3] { new PointF(foot.X - length / 8, foot.Y + length / 8), new PointF(foot.X, foot.Y + length / 8), new PointF(foot.X + length / 8, foot.Y + length / 8) };
                        legs.Add(new Leg(spineconnection, knee, foot, toes, length));
                    }
                }
                return legs;
            }
            double cyclepoint;
            double swingcycle = 0;
            bool swinging = false;
            public PointF swingdestination;
            double CalculateAngleDifference(double angle1, double angle2)
            {
                return Math.Atan2(Math.Sin(angle2-angle1), Math.Cos(angle2-angle1));
                double diff = angle1 - angle2;
                diff = (diff + Math.PI) % (2 * Math.PI) - Math.PI;
                return Math.Abs(diff);
            }
            public void WalkCycle(PointF newspine, float angle, double speed)
            {
                //Assume foot starts futherest foward for cyclepoint % 2PI == 0
                while (cyclepoint >= Math.PI * 2)
                {
                    cyclepoint -= Math.PI * 2;
                }

                if (swinging)
                {
                    swingdestination = new PointF(startfoot.X + (length * 0.75f) * MathF.Cos(angle), startfoot.Y + (length * 0.75f) * MathF.Sin(angle));
                    var newfoot = DragPoint(foot, swingdestination, speed * 6, 1);

                    for (int i = 0; i < toes.Count(); ++i)
                    {
                        toes[i] = new PointF(toes[i].X + newfoot.X - foot.X, toes[i].Y + newfoot.Y - foot.Y);
                        continue;
                    }
                    foot = newfoot;

                    //foot = swingdestination;
                    if (foot.DistanceTo(swingdestination) < 1)
                    {
                        swinging = false; //Let this be the new anchor
                    }
                }
                //Completely ignore the knee for now
                else if (spineconnection.DistanceTo(foot) > length * 1.25)
                {
                    //Start swinging foot foward
                    swinging = true;

                    //Set the default values for the feet
                    var centre_toe = new PointF((float)Math.Cos(angle), (float)Math.Sin(angle)).UnitVector();
                    toe_start[0] = new PointF(centre_toe.X * toelength + foot.X, centre_toe.Y * toelength + foot.Y);

                    var left_toe = new PointF(
                        (float)(centre_toe.X * Math.Cos(Math.PI / 6) - centre_toe.Y * Math.Sin(Math.PI / 6)),
                        (float)(centre_toe.X * Math.Sin(Math.PI / 6) + centre_toe.Y * Math.Cos(Math.PI / 6))
                    );
                    toe_start[1] = new PointF(left_toe.X * toelength + foot.X, left_toe.Y * toelength + foot.Y);

                    // Rotate the vector by -30 degrees (counterclockwise)
                    var right_toe = new PointF(
                        (float)(centre_toe.X * Math.Cos(-Math.PI / 6) - centre_toe.Y * Math.Sin(-Math.PI / 6)),
                        (float)(centre_toe.X * Math.Sin(-Math.PI / 6) + centre_toe.Y * Math.Cos(-Math.PI / 6))
                    );
                    toe_start[2] = new PointF(right_toe.X * toelength + foot.X, right_toe.Y * toelength + foot.Y);

                    for (int i = 0; i < 3; ++i)
                    {
                        toes[i] = toe_start[i]; //Holy shit this is inefficient
                    }
                }
                else
                {
                    //Rotate the toes
                    int direction = 1;
                    double ft = CalculateAngle(foot, toes[1]);
                    double fk = CalculateAngle(foot, knee);

                    // Calculate the shortest distance between the angles
                    double distance = CalculateAngleDifference(ft, fk);

                    // Check if increasing `ft` by 180 degrees would intersect `fk`
                    bool intersects = (CalculateAngleDifference(ft + Math.PI, fk) < distance) && (ft < fk);
                    if (intersects) //Would increasing out angle by 180 degrees intersect with fk?
                    {
                        direction = -1;
                    }
                    else
                    {
                        direction = 1;
                    }

                    for (int i = 0; i < toes.Count(); ++i)
                    {
                        var toe = toes[i];

                        var ft_angle = CalculateAngle(foot, toe);

                        // Determine the direction to rotate ft_angle away from kf_angle
                        ft_angle += speed * 0.01 * direction;

                        // Calculate the new position of the toe using the new angle
                        var newX = foot.X + (float)(toelength * Math.Cos(ft_angle));
                        var newY = foot.Y + (float)(toelength * Math.Sin(ft_angle));

                        // Update toe to the new position
                        toes[i] = new PointF(newX, newY);
                    }
                }

                var legcentre = new PointF((foot.X + spineconnection.X) / 2f, (foot.Y + spineconnection.Y) / 2f);
                var legsize = foot.DistanceTo(spineconnection);
                knee = legcentre;
                double desiredlength = length * 1.3;

                if (legsize < desiredlength)
                {
                    var perpindicular = PerpendicularVector(foot, spineconnection);

                    float adjustment = (float)(Math.Sqrt(desiredlength * desiredlength - legsize * legsize) / 2.0);
                    knee.X += MathF.Cos(angle) * adjustment;
                    knee.Y += MathF.Sin(angle) * adjustment;
                }
                spineconnection = newspine;
                return;
            }
            public static PointF PerpendicularVector(PointF p0, PointF p1)
            {
                float dx = p1.X - p0.X;
                float dy = p1.Y - p0.Y;

                // Calculate the perpendicular vector
                PointF perpendicular = new PointF(-dy, dx);

                // Calculate the magnitude of the perpendicular vector
                float magnitude = (float)Math.Sqrt(perpendicular.X * perpendicular.X + perpendicular.Y * perpendicular.Y);

                // Normalize the vector to make it a unit vector
                return new PointF(perpendicular.X / magnitude, perpendicular.Y / magnitude);
            }

        }
    }
}