using System.Runtime.InteropServices;
using Microsoft.VisualBasic;

public unsafe class Map
{
    int* gridSquares;
    public int width;
    public int height;
    public Map(int width, int height)
    {
        this.width = width;
        this.height = height;
        fixed (int* arry = new int[width * height])
        {
            gridSquares = arry;
        }
    }
    public unsafe int* GetGridSquare(int x, int y)
    {
        if (x >= width || y >= height)
        {
            throw new IndexOutOfRangeException("x or y out of range for GetGridSquare");
        }
        return &gridSquares[x + y * width];
    }
    List<Room> rooms = new List<Room>();

    public void GenerateMap()
    {
        Random r = new Random();
        //Generate a group of rooms
        int maxrooms = 4;
        int builtrooms = 0;
        //Rooms can be from 4x4 to 10x10
        int xlocation;
        int ylocation;
        while (builtrooms < maxrooms)
        {
            int roomwidth = r.Next(4, 11);
            int roomheight = r.Next(4, 11);
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
                    GetGridSquare(x, y)[0] = (int)GridSquareType.Floor;
                }
            }
        }
        //Connect rooms via hallways
        //Assign doorways to each of the rooms, disallow any other entrances
        foreach (var room in rooms)
        {
            Point random_top_entrance = new Point(room.bounds.X + r.Next(0, room.bounds.Width), room.bounds.Y);
            Point random_bottom_entrance = new Point(room.bounds.X + r.Next(0, room.bounds.Width), room.bounds.Y + room.bounds.Height);
            Point random_left_entrance = new Point(room.bounds.X, room.bounds.Y + r.Next(0, room.bounds.Height));
            Point random_right_entrance = new Point(room.bounds.X + room.bounds.Width, room.bounds.Y + r.Next(0, room.bounds.Height));
            room.doors.Add(random_top_entrance);
            room.doors.Add(random_bottom_entrance);
            room.doors.Add(random_left_entrance);
            room.doors.Add(random_right_entrance);

            foreach(var door in room.doors)
            {
                GenerateMazeFromPoint(door.X + door.Y * width);
            }
        }
    }
    public void GenerateMazeFromPoint(int p)
    {
        Random r = new Random();
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
                    lastdirection = direction;
                    gridSquares[p] = (int)GridSquareType.Floor;
                    placed = true;
                }
            } while (directions.Count() >= 1);
            if (!placed)
            {
                canmove = false;
            }
        }
    }
    public bool CanMove(int location, int direction)
    {
        int x = location % width;
        int y = location / width;
        //Check to make sure not on edge
        if (Math.Abs(direction) == 1)
        {
            if (x + direction == -1 || x + direction == width)
            {
                return false;
            }
        }
        else
        {
            if (y + direction == -1 || y + direction == height)
            {
                return false;
            }
        }
        if (*(gridSquares + location + direction) == (int)GridSquareType.Wall)
        {
            //Check all the squares nearby
            int abovesquare = location + direction;

            int upup = -1;
            int upleft = -1;
            int upright = -1;

            if (Math.Abs(direction) == 1) //Moving left/right
            {
                if (y < width - 1)
                {
                    upleft = abovesquare + width;
                }
                if (y >= 1)
                {
                    upright = abovesquare - width;
                }
                if (x + direction + direction >= 1 && x + direction + direction < width - 1)
                {
                    upup = abovesquare + direction;
                }
            }
            else
            {
                if (x > 1)
                {
                    upright = abovesquare - 1;
                }
                if (x < width - 1)
                {
                    upleft = abovesquare + 1;
                }
                if (y + direction + direction >= 1 && y + direction + direction < height - 1)
                {
                    upup = abovesquare + direction;
                }
            }

            bool result = true;
            if (upup != -1)
            {
                if (*(gridSquares + upup) == (int)GridSquareType.Floor) //Dont draw two floors next to each other
                {
                    result = false;
                }
            }
            if (upleft != -1)
            {
                if (*(gridSquares + upleft) == (int)GridSquareType.Floor) //Dont draw two floors next to each other
                {
                    result = false;
                }
            }
            if (upright != -1)
            {
                if (*(gridSquares + upright) == (int)GridSquareType.Floor) //Dont draw two floors next to each other
                {
                    result = false;
                }
            }
            return result;
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
}