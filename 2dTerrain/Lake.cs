﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Security.Cryptography.Xml;
using System.Text;
using System.Threading.Tasks;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace TerrainGenerator
{
    public class Lake
    {
        public List<Point> bounds = new List<Point>();
        public Lake()
        {

        }
        public struct Bump
        {
            public double degrees;
            public double radius;

            public Bump(double degrees, double radius)
            {
                this.degrees = degrees;
                this.radius = radius;
            }
        }
        const double MAX_BUMP_SIZE = 0.1; //Dont let a bump be more than 0.4x the 
        public static Lake GenerateLake(Rectangle bounds, int seed=-1)
        {
            Lake result = new Lake();
            Random r = seed == -1 ? new Random() : new Random(seed); //Assign with seed if it is available, otherwise make it completely randmo
            //Generate some lakeish shape inside the bounds
            int bumps = 1; //Lake by default is a circle, this changes the amount of bumps on the side of the lake
            bumps = r.Next(8,10);
            List<Bump> bumpdegrees = new List<Bump>();
            for(int i = 0; i < bumps; i++)
            {
                //Create n bumps inside the list at a random degrees
                double radius = r.NextDouble() * Math.Sqrt((bounds.Width * bounds.Height) * MAX_BUMP_SIZE); //Find the max area, then get the sqrt to find the max area
                bumpdegrees.Add(new Bump(r.NextDouble() * 360, radius));
            }

            for(double i = 0; i < 360; i+=1)
            {
                double angleInRadians = i * (Math.PI / 180.0);

                //Ellipse equation is (x^2)/(a^2) + (y^2)/(b^2) = 1
                //The parametric equations for an ellipse are given by:

                //x(θ) = acos(θ)
                //y(θ) = bsin(θ)

                double x = (bounds.Width / 2) * Math.Cos(angleInRadians);
                double y = (bounds.Height / 2) * Math.Sin(angleInRadians);
                double radius = Math.Sqrt(x * x + y * y);

                //Check for bumps
                foreach (Bump b in bumpdegrees)
                {
                    var b_radians = b.degrees * (Math.PI / 180.0);
                    double distance = AngleCartesianDistance(angleInRadians, b_radians, bounds);
                    if (distance < b.radius*4) //Inside the influence of the bump
                    {
                        //Imagine a new rectangle for a new ellipse to be drawn
                        //Rectangle will be on an angle
                        //First point of y distance radius, x distance radius
                        double maxratio = ((radius + b.radius) / radius);
                        PointF maxradius_point = new PointF((float)(maxratio * Math.Cos(b_radians) * (bounds.Width / 2)), (float)(maxratio * Math.Sin(b_radians) * (bounds.Height / 2)));


                        //Increase the new radius by an inverse scale of the bumps radius
                        //For distance = 0, radius = currentradius + b.radius
                        //For distance = b.radius, radius = currentradius

                        

                        double newradius = radius + RockSmootheCurve(distance, b.radius*4, b.radius/2);
                        double ratio = newradius / radius;
                        x *= ratio;
                        y *= ratio; //find the ratio difference between the radius's and multiply the new points by that value
                    }
                }

                result.bounds.Add(new Point((int)x + bounds.X + bounds.Width/2, (int)y + bounds.Y + bounds.Height/2)); //Adjust the points from relative to cartesian (0,0) to the box
            }
            return result;
        }
        public static double RockSmootheCurve(double distance, double xcutoff, double ycutoff)
        {
            //distance = x, xcutoff = r, ycutoff = c
            //y = ((c) / (r ^ (2)))(-x ^ (2) + r ^ (2))
            return (ycutoff / (xcutoff * xcutoff)) * (-distance*distance + xcutoff*xcutoff);
        }
        private static double AngleCartesianDistance(double angleInRadians, double b_radians, Rectangle bounds)
        {
            return Math.Sqrt(Math.Pow((Math.Sin(b_radians) - Math.Sin(angleInRadians)) * bounds.Height, 2) + Math.Pow((Math.Cos(b_radians) - Math.Cos(angleInRadians)) * bounds.Width, 2));
        }
    }
}