using System.Runtime.InteropServices;
using Microsoft.VisualBasic;
namespace TerrainGenerator
{

    public unsafe class Map
    {
        private class Path
        {
            public List<Point> points = new List<Point>();
            public bool linked;
            public Path(List<Point> points)
            {
                this.points = points;
            }
        }
        int[] gridSquares;
        public int width;
        public int height;
        public Map(int width, int height)
        {
            this.width = width;
            this.height = height;
            gridSquares = new int[width * height];
        }
        public unsafe int GetGridSquare(int x, int y)
        {
            if (x >= width || y >= height)
            {
                throw new IndexOutOfRangeException("x or y out of range for GetGridSquare");
            }
            return gridSquares[x + y * width];
        }
        public unsafe void SetGridSquare(int x, int y, int value)
        {
            if (x >= width || y >= height)
            {
                throw new IndexOutOfRangeException("x or y out of range for GetGridSquare");
            }
            gridSquares[x + y * width] = value;
        }
        List<Room> rooms = new List<Room>();

        public void GenerateMap()
        {
            Random r = new Random();
            //Generate a group of rooms
            const int maxrooms = 50;
            const int minsize = 4;
            const int maxsize = 20;

            int builtrooms = 0;
            //Rooms can be from 4x4 to 10x10
            int xlocation;
            int ylocation;
            while (builtrooms < maxrooms)
            {
                int roomwidth = r.Next(minsize, maxsize);
                int roomheight = r.Next(minsize, maxsize);
                xlocation = r.Next(0, width - roomwidth);
                ylocation = r.Next(0, height - roomheight);
                //Check if we intersect with any other rooms
                Room room = new Room(new Rectangle(xlocation, ylocation, roomwidth, roomheight));
                if (RectangleIntersects(room.bounds, rooms.Select(r => r.bounds).ToList()))
                {
                    continue; //Dont have overlapping rooms
                }
                rooms.Add(room);
                builtrooms++;
                //Clear out the whole room location
                for (int x = xlocation; x < xlocation + roomwidth; ++x)
                {
                    for (int y = ylocation; y < ylocation + roomheight; ++y)
                    {
                        SetGridSquare(x, y, (int)GridSquareType.Floor);
                    }
                }
            }
            //Connect rooms via hallways
            //Assign doorways to each of the rooms, disallow any other entrances
            List<Path> paths = new List<Path>();

            foreach (var room in rooms)
            {
                int doorstoremove = r.Next(0, 4);
                Point random_top_entrance = new Point(room.bounds.X + r.Next(0, room.bounds.Width), room.bounds.Y);
                Point random_bottom_entrance = new Point(room.bounds.X + r.Next(0, room.bounds.Width), room.bounds.Y + room.bounds.Height);
                Point random_left_entrance = new Point(room.bounds.X, room.bounds.Y + r.Next(0, room.bounds.Height));
                Point random_right_entrance = new Point(room.bounds.X + room.bounds.Width, room.bounds.Y + r.Next(0, room.bounds.Height));

                room.doors.Add(random_top_entrance);
                room.doors.Add(random_bottom_entrance);
                room.doors.Add(random_left_entrance);
                room.doors.Add(random_right_entrance);

                //Remove doors
                for (int i = 0; i < doorstoremove; ++i)
                {
                    //room.doors.RemoveAt(r.Next(0, room.doors.Count));
                }

                foreach (var door in room.doors)
                {
                    var toadd = new Path(GenerateMazeFromPoint(door.X + door.Y * width));
                    toadd.points.Add(door);
                    paths.Add(toadd);
                }
            }

            foreach (var path in paths)
            {
                List<Point> added = new List<Point>();
                foreach (Point p in path.points)
                {
                    foreach (var room in rooms)
                    {
                        if (!room.bounds.Contains(p) && GetGridSquare(p.X, p.Y) == 1)
                        {
                            added.AddRange(GenerateMazeFromPoint(p.X + p.Y * width));
                        }
                    }
                }
                path.points.AddRange(added);
            }

            //Look for paths that are seperated by one
            for (int i = 0; i < paths.Count; ++i)
            {
                var path = paths[i];
                if (!path.linked)
                {
                    foreach (var p1 in path.points)
                    {
                        if (path.linked)
                        {
                            break;
                        }
                        for (int j = 0; j < paths.Count; j++)
                        {
                            Path? path2 = paths[j];
                            if (j != i)
                            {
                                var closestpoints = path2.points.Where(p2 => Math.Abs(p2.X - p1.X) == 2 || Math.Abs(p2.Y - p1.Y) == 2).ToList();
                                if (closestpoints.Count() >= 1) //Exactly one space between them
                                {
                                    var closestpoint = closestpoints[0];
                                    if (Math.Abs(closestpoint.X - p1.X) == 2) //2 apart on x axis
                                    {
                                        //free upthe space inbetween
                                        SetGridSquare((closestpoint.X + p1.X) / 2, p1.Y, (int)GridSquareType.Floor);
                                    }
                                    else //2 apart on y axis
                                    {
                                        SetGridSquare(p1.X, (p1.Y + closestpoint.Y) / 2, (int)GridSquareType.Floor);
                                    }
                                    //Joined path, exit
                                    path.linked = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        public List<Point> GenerateMazeFromPoint(int p)
        {
            List<Point> result = new List<Point>();
            bool canmove = true;
            int lastdirection = 0;
            while (canmove)
            {
                Stack<int> directions = new Stack<int>((new int[] { 1, -1, width, -width }).ToList());
                if (lastdirection != 0)
                {
                    directions.Push(lastdirection); //Weight going in the same direction
                }
                directions = directions.Shuffle();
                bool placed = false;
                do
                {
                    var direction = directions.Pop();
                    if (CanMove(p, direction))
                    {
                        p += direction;
                        if (p < 0 || p >= width * height)
                        {
                            MessageBox.Show("Out of range?");
                        }
                        lastdirection = direction;
                        gridSquares[p] = (int)GridSquareType.Floor;
                        result.Add(new Point(p % width, p / width));
                        placed = true;
                    }
                } while (directions.Count() >= 1);
                if (!placed)
                {
                    canmove = false;
                }
            }
            return result;
        }
        public bool CanMove(int location, int direction, bool forcejoin = false)
        {
            int x = location % width;
            int y = location / width;
            if (x == 0 || x == width - 1 || y == 0 || y == height - 1) //Dont check edges
            {
                return false;
            }

            //Check to make sure not on edge
            if (Math.Abs(direction) == 1)
            {
                if (x + direction < 1 || x + direction >= width - 1)
                {
                    return false;
                }
            }
            else
            {
                if (y + direction / width < 1 || y + direction / width >= height - 1)
                {
                    return false;
                }
            }
            if (forcejoin)
            {
                return true;
            }
            if (gridSquares[location + direction] == (int)GridSquareType.Wall)
            {
                //Check all the squares nearby
                int abovesquare = location + direction;
                int newx = abovesquare % width;
                int newy = abovesquare / width;
                if (newx == 0 || newy == 0 || newx == width - 1 || newy == height - 1)
                {
                    string proof = $"Direction: {direction}\n({x},{y})\n{x} + {direction} < 0 || {x} + {direction} >= {width - 1}\n{y} + {direction / width} < 0 || {y} + {direction / width} >= {height - 1}";
                    MessageBox.Show(proof);
                }

                //Check the contents of every pixel around above square
                //We can assume no bounds checks are nessecary?
                bool upleft;
                bool up;
                bool upright;
                bool left;
                bool leftdown;
                bool down;
                bool downright;
                bool right;
                try
                {
                    upleft = gridSquares[abovesquare - width - 1] == 0;
                    up = gridSquares[abovesquare - width] == 0;
                    upright = gridSquares[abovesquare - width + 1] == 0;

                    left = gridSquares[abovesquare - 1] == 0;
                    leftdown = gridSquares[abovesquare + width - 1] == 0;
                    down = gridSquares[abovesquare + width] == 0;
                    downright = gridSquares[abovesquare + width + 1] == 0;
                    right = gridSquares[abovesquare + 1] == 0;
                }
                catch
                {
                    MessageBox.Show(abovesquare.ToString());
                    throw new Exception("Lol what");
                }
                if (direction == 1)
                {
                    //Moving right?
                    //Mark left of next pixel as available
                    left = true;
                    upleft = true;
                    leftdown = true;
                }
                else if (direction == -1)
                {
                    //Moving left?
                    //Mark right of next pixel as available
                    right = true;
                    upright = true;
                    downright = true;
                }
                else if (direction == width)
                {
                    //Moving down?
                    //Mark up of next pixel as available
                    up = true;
                    upright = true;
                    upleft = true;
                }
                else if (direction == -width)
                {
                    //Moving up?
                    //Mark down of next pixel as available
                    down = true;
                    downright = true;
                    leftdown = true;
                }

                return upleft && up && upright && left && leftdown && down && downright && right;
            }
            return false;
        }
        public static bool RectangleIntersects(Rectangle rectangle, List<Rectangle> rectangles)
        {
            foreach (var r in rectangles)
            {
                //Check for all corners in my rectangle and see if they are inside the other rectangle
                if (PointInRectangle(new Point(rectangle.X, rectangle.Y), r)) //Top left corner
                {
                    return true;
                }
                if (PointInRectangle(new Point(rectangle.X + rectangle.Width, rectangle.Y), r)) //Top left corner
                {
                    return true;
                }
                if (PointInRectangle(new Point(rectangle.X, rectangle.Y + rectangle.Height), r)) //Top left corner
                {
                    return true;
                }
                if (PointInRectangle(new Point(rectangle.X + rectangle.Width, rectangle.Y + rectangle.Height), r)) //Top left corner
                {
                    return true;
                }
            }
            //Nothing intersected?
            return false;
        }
        public static bool PointInRectangle(Point p, Rectangle r) //Also checks if point is on edge
        {
            if (p.X >= r.X && p.X <= r.X + r.Width && p.Y >= r.Y && p.Y <= r.Y + r.Height)
            {
                return true;
            }
            return false;
        }
    }
    public class Room
    {
        public Rectangle bounds;
        public List<Point> doors = new List<Point>();
        public Room(Rectangle bounds)
        {
            this.bounds = bounds;
        }
    }
    public enum GridSquareType
    {
        Wall = 0,
        Floor = 1,
        Marked = 2,
    }
}