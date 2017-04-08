import java.util.*;

public class Cluster {
	int Label;
    List<RGBVector> pixels;
    RGBVector centroid;
    
	public Cluster(int label) {
		// TODO Auto-generated constructor stub
	 
		this.Label= label;
		this.pixels = new ArrayList<RGBVector>();
	}
	
	public void addPixel(RGBVector point)
	{
		this.pixels.add(point);
	}
	
	public void clustInfo() {
		System.out.println("Cluster: " + this.Label);
		System.out.println("Centroid: " + this.centroid);
		System.out.println("[Plots are: \n");
		for(RGBVector p : pixels) {
			System.out.println(p);
		}
		System.out.println("]");
	}
 
	
	public void reset() {
		pixels.clear();
	}

}
