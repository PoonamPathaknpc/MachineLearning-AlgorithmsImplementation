/***
 * Assignment 3 ML 
 * Poonam Pathak
*****/

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javax.imageio.ImageIO;
 

public class Kmeans {
	
	public static List<Cluster> clusters = new ArrayList<Cluster>();
	public static List<RGBVector> pixels = new ArrayList<RGBVector>();
	public static int[] rgb;
	
	public Kmeans(){		
		
	}
	
    public static void main(String [] args){
    	
    int[] Kvalue = {2,5,10,15,20};     
	
	
    try{
		File f = new File(".\\Images");
		File[] imgfiles = f.listFiles();
		for(int h=0;h<imgfiles.length;h++)
		{
		  if(imgfiles[h].isFile())	
		   {
			  String name = imgfiles[h].getName();  
		  	  File orignalFile =  new File(f.getCanonicalPath() + "\\" + name);	
		      System.out.println("Compression results for image: " + orignalFile);
	          BufferedImage originalImage = ImageIO.read(orignalFile);	      
	          long OrigSize = (orignalFile.length())/1024;
	          // For different values of K...
	      
	          for(int k=0;k<Kvalue.length;k++)
	          {	      	
	    	    clusters.clear();
		        pixels.clear();	
	            BufferedImage kmeansJpg = kmeans_helper(originalImage,Kvalue[k]);	        
	            File fnew = new File(f.getCanonicalPath() + "\\" + name.split("\\.")[0]);
	            fnew.mkdir();
	            ImageIO.write(kmeansJpg, "jpg", new File(fnew.getCanonicalPath() + "\\" + name.split("\\.")[0] + "K" + Kvalue[k]+ ".jpg")); 
	            File newImage = new File(fnew.getCanonicalPath() + "\\" + name.split("\\.")[0] + "K" + Kvalue[k] + ".jpg");
	        
	            // Calculating the Compression ratio...
	            long Compsize = newImage.length()/1024;	        
	            long compratio = OrigSize/Compsize;
	        
	            System.out.println("For K: " + Kvalue[k]);
	            System.out.println("The CompressedImage generated is : " + newImage.getName());
	            System.out.println("Compression ratio : "  + compratio);
	            System.out.println();
	    
	           }
	      
	          //need to calculate variance and average of compression ratios..
	          
	      System.out.println();
	      System.out.println("*****************************************************************************************");
		  }
		  
		}
	   }catch(IOException e){
	    System.out.println(e.getMessage() + " test");
	    }	
    }
    
    private static BufferedImage kmeans_helper(BufferedImage originalImage, int k){
	int w=originalImage.getWidth();
	int h=originalImage.getHeight();
	BufferedImage kmeansImage = new BufferedImage(w,h,originalImage.getType());
	Graphics2D g = kmeansImage.createGraphics();
	g.drawImage(originalImage, 0, 0, w,h , null);	
	
	// Read rgb values from the image
	rgb=new int[w*h];
	int count=0;
	for(int i=0;i<w;i++){
	    for(int j=0;j<h;j++){	    	
		rgb[count++]=kmeansImage.getRGB(i,j);
			
	    }
	}
	// Call kmeans algorithm: update the rgb values
	kmeans(k,w,h);

	// Write the new rgb values to the image
	count=0;
	for(int i=0;i<w;i++){
	    for(int j=0;j<h;j++){
	    	
		kmeansImage.setRGB(i,j,rgb[count++]);
	    }
	}
	return kmeansImage;
    }

    
    
    // k-means Clustering code with given value of K...    
    private static void kmeans(int k, int w, int h){
    	
    	boolean done = false;    	
    	int count=0;
		for(int i=0;i<w;i++)
		{			
		  for(int j=0;j<h;j++)
		  {
    		int ARGB = rgb[count++];
    		
    		// Fetching the RGB components of the pixel
    		float blue = ARGB & 0xff;
    		float green = (ARGB & 0xff00) >> 8;
    		float red = (ARGB & 0xff0000) >> 16;
	
	        // creating pixel Array list for k means clustering..
	        RGBVector pixel = new RGBVector(red, green, blue);	        
	        pixels.add(pixel);
    	  }
		}
    	
	        // Initializing the k clusters with random pixel assignment..    
	        for(int i=0;i<k;i++)
			{
	        	
				Cluster cluster = new Cluster(i);
				Random rand = new Random();
				int r = rand.nextInt(pixels.size()-1);			
				cluster.centroid = pixels.get(r);
				clusters.add(cluster);
				
			}
			
			// Iteratively calculating and recalculating the clusters
			//Setting Iteration to a value = 10 
	        // int iteration=10;
	        double distance1=0;
	        double gap1=0;
	        
			while(!done) {
				
	        	//Clear cluster state
	        	clearClusters();
	        	
	        	List<RGBVector> lastCen = CentroidList();
	        	
	        	//Assign points to the closer cluster
	        	ClusterReassignment();
	            
	            //Calculate new centroids based on the rearrangements of the RGB vectors in clusters
	        	ReCalCentroids(); 	        	
	        	List<RGBVector> newCen = CentroidList();
	        	
	        	//Calculates total distance between new and old Centroids
	        	double distance2 = 0;	        	
	        	for(int i = 0; i < lastCen.size(); i++) {
	        		//System.out.println("for: cluster  " + i);
	        		//System.out.println(lastCen.get(i).B + " " +  lastCen.get(i).G + " " + lastCen.get(i).R);
	        		//System.out.println(lastCen.get(i).B + " " + newCen.get(i).G + " " + newCen.get(i).R);
	        		distance2+= RGBVector.CalcVectorDist(lastCen.get(i),newCen.get(i));
	        		
	        	}
	        	//System.out.println(distance2 + " " +  distance1);
	        	double gap2 = distance2-distance1;
	        	//System.out.println(gap2);
	        	if(gap2-gap1 == 0) {
	        		done = true;
	        	}
	        	else 
	        	    {
	        		distance1=distance2;
	        	    gap1=gap2;
	        	    }	        	
		   }
			
		
	        count=0;
	       
	        for(RGBVector pixel : pixels)
			{
	        	    int clustn = pixel.cluster_num;
	        	    Cluster cluster = clusters.get(clustn);   	    
                    //System.out.println("point : " + cluster.centroid.R + " G : "  + cluster.centroid.G + " B: " + cluster.centroid.B);
	        	    int rgbpoint = (Math.round(cluster.centroid.B )) | (Math.round(cluster.centroid.G) << 8) | (Math.round(cluster.centroid.R) << 16); 
			    	rgb[count++] = rgbpoint;
			    	
			 }
		
    }
    
    
    
	  private static void clearClusters() {
	    	for(Cluster cluster : clusters) {
	    		cluster.reset();
	    	}
	    }
	    
	    private static List<RGBVector> CentroidList() {
	    	List<RGBVector> centroids = new ArrayList<RGBVector>();
	    	for(Cluster cluster : clusters) {
	    		RGBVector point = cluster.centroid;		    		
	    		centroids.add(point);
	    	}
	    	return centroids;
	    }
	    
	    private static void ClusterReassignment() {
	        double max = Double.MAX_VALUE;
	        double min = max; 
	        int cluster = 0;                 
	        double distance = 0.0; 
	        
	        
	        for(RGBVector pixel : pixels) {
	        	min = max;
	            for(int i = 0; i<clusters.size(); i++) {
	            	Cluster c = clusters.get(i);	            	
	                distance = RGBVector.CalcVectorDist(pixel, c.centroid);
	                if(distance < min){
	                    min = distance;
	                    cluster = i;
	                }
	            }
	            pixel.cluster_num = cluster;
	            clusters.get(cluster).addPixel(pixel);
	        }
	    }
	    
	    private static void ReCalCentroids() {
	        for(Cluster cluster : clusters) {
	            float red = 0;
	            float green = 0;
	            float blue = 0;
	            List<RGBVector> list = cluster.pixels;
	            int size = list.size();
	            
	            for(RGBVector pixel : list) {
	            	red += pixel.R;
	            	blue += pixel.B;
	                green += pixel.G;
	            }
	            
	            RGBVector centroid = cluster.centroid;
	            
	            if(size > 0) {
	            	float R = red / size;
	                float G = green / size;
	                float B = blue /size;
	                centroid.R = R;
	                centroid.G = G;
	                centroid.B = B;
	            }
	           
	        }
	    }


}