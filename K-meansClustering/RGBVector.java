import java.math.BigDecimal;
import java.math.RoundingMode;

public class RGBVector {

	Float R;
	Float G;
	Float B;
	int cluster_num=-1;
	
	public RGBVector() {}
	
	public RGBVector(Float r , Float g, Float b) {
		R = r;
		G= g;
		B = b;
		
	}
	
	protected static double CalcVectorDist(RGBVector point, RGBVector centroid) {
		
		BigDecimal R = new BigDecimal(Math.pow((double)centroid.R - point.R, 2));
		BigDecimal G = new BigDecimal(Math.pow((double)centroid.G - point.G, 2));
	    BigDecimal B = new BigDecimal(Math.pow((double)centroid.B - point.B,2));        
        R = R.add(G).add(B);
        R.setScale(2, RoundingMode.HALF_DOWN);
        double dist = Math.sqrt(R.doubleValue());
        return dist;
    }
 

}
