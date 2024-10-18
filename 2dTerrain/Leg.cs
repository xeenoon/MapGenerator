namespace TerrainGenerator
{

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
        public Leg(PointF spineconnection, PointF kee, PointF foot, int length)
        {
            this.spineconnection = spineconnection;
            this.knee = kee;
            this.foot = foot;
            this.length = length;
            toelength = length / 6;
            this.toes = new PointF[3];

            startfoot = foot;
        }
        public static Leg AddLeg(double angle, PointF spineconnection)
        {

            double xdirection = Math.Cos(angle);
            double ydirection = Math.Sin(angle);
            int leglength = sectionwidth * 5;

            PointF knee = new PointF((float)(spineconnection.X + xdirection * leglength / 2.0),
            (float)(spineconnection.Y + ydirection * leglength / 2.0));

            PointF foot = new PointF((float)(spineconnection.X + xdirection * leglength),
            (float)(spineconnection.Y + ydirection * leglength));

            Leg result = new Leg(spineconnection, knee, foot, leglength);
            //Set the default values for the feet
            var centre_toe = new PointF((float)Math.Cos(angle), (float)Math.Sin(angle)).UnitVector();
            result.toe_start[0] = new PointF(centre_toe.X * result.toelength + foot.X, centre_toe.Y * result.toelength + foot.Y);

            var left_toe = new PointF(
                (float)(centre_toe.X * Math.Cos(Math.PI / 6) - centre_toe.Y * Math.Sin(Math.PI / 6)),
                (float)(centre_toe.X * Math.Sin(Math.PI / 6) + centre_toe.Y * Math.Cos(Math.PI / 6))
            );
            result.toe_start[1] = new PointF(left_toe.X * result.toelength + foot.X, left_toe.Y * result.toelength + foot.Y);

            // Rotate the vector by -30 degrees (counterclockwise)
            var right_toe = new PointF(
                (float)(centre_toe.X * Math.Cos(-Math.PI / 6) - centre_toe.Y * Math.Sin(-Math.PI / 6)),
                (float)(centre_toe.X * Math.Sin(-Math.PI / 6) + centre_toe.Y * Math.Cos(-Math.PI / 6))
            );
            result.toe_start[2] = new PointF(right_toe.X * result.toelength + foot.X, right_toe.Y * result.toelength + foot.Y);

            for (int i = 0; i < 3; ++i)
            {
                result.toes[i] = result.toe_start[i]; //Holy shit this is inefficient
            }
            return result;
        }
        public const int LEGDST = 4;
        public static List<Leg> BuildLegs(PointF[] spine, int length)
        {
            List<Leg> legs = new List<Leg>();
            for (int i = 1; i < spine.Length / LEGDST; ++i) //Start at one to not draw a leg on the head
            {
                PointF spineconnection = spine[i * LEGDST];

                //Assume spine[] y's are all the same
                for (int direction = -1; direction < 2; direction += 2)
                {
                    PointF knee = new PointF(spineconnection.X, spineconnection.Y + direction * length / 2);
                    PointF foot = new PointF(spineconnection.X, spineconnection.Y + direction * length);
                    legs.Add(new Leg(spineconnection, knee, foot, length));
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
            return Math.Atan2(Math.Sin(angle2 - angle1), Math.Cos(angle2 - angle1));
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
