using System.Runtime.InteropServices;
using TerrainGenerator;

public unsafe class Filter
{
    public int r;
    public int g;
    public int b;
    public Filter(int r, int g, int b)
    {
        this.r = r;
        this.g = g;
        this.b = b;
    }
    public void ApplyFilter(byte* colordata)
    {
        colordata[0] = (byte)Math.Min(255, b + colordata[0]);
        colordata[1] = (byte)Math.Min(255, g + colordata[1]);
        colordata[2] = (byte)Math.Min(255, r + colordata[2]);
    }
    public int* GetArry()
    {
        int* result = (int*)Marshal.AllocHGlobal(3 * sizeof(int));
        result[0] = b;
        result[1] = g;
        result[2] = r;
        return result;
    }
    public enum RockTintType
    {
        Red,
        Dark,
        Light,
        Yellow,
    }
    const int variance = 20;
    public static Filter RandomFilter()
    {
        //Not quite random, designed off of different rock tints
        Random r = new Random();
        var tinttype = (RockTintType)r.Next(0, 4);
        Filter result = new Filter(0, 0, 0);

        switch (tinttype)
        {
            case RockTintType.Red:
                result.r = r.Next(0, variance);
                break;
            case RockTintType.Dark:
                int darkness = r.Next(0, variance);
                result.r = -darkness;
                result.g = -darkness;
                result.b = -darkness;
                break;
            case RockTintType.Light:
                int brightness = r.Next(0, variance);
                result.r = brightness;
                result.g = brightness;
                result.b = brightness;
                break;
            case RockTintType.Yellow:
                int yellowness = r.Next(0, variance);

                result.r = yellowness;
                result.g = yellowness; //Yellow = red + green
                break;
        }
        return result;
    }
}