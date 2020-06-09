import gcrfs.algorithms.Basic;
import gcrfs.algorithms.DirGCRF;
import gcrfs.calculations.CalculationsDirGCRF;
import gcrfs.data.datasets.Dataset;
import gcrfs.data.generators.ArrayGenerator;
import gcrfs.data.generators.GraphGenerator;
import gcrfs.learning.GradientAscent;
import gcrfs.learning.Parameters;

public class Main {

	public static void main(String[] args) {
		double[][] s = GraphGenerator.generateDirectedGraph(200);
		double[] r = ArrayGenerator.generateArray(200, 5);
		
		CalculationsDirGCRF c = new CalculationsDirGCRF(s, r);
		double[] y = c.y(1, 2, 0.05);
		Dataset dataset = new Dataset(s, r, y);
		
		double alpha = 1;
		double beta = 1;
		double lr = 0.0001;
		int maxIter = 100;
		Parameters p = new Parameters(alpha, beta, maxIter, lr, false, null);
				
		// learning algorithm
		GradientAscent g = new GradientAscent(p, c, y);
				
		Basic method = new Basic(g, c, dataset);
				
		double[] predictedOutputs = method.predictOutputs();
		for (int i = 0; i < predictedOutputs.length; i++) {
			System.out.println(predictedOutputs[i]);
		}
				
		System.out.println("R^2 Train: " + method.rSquared());
	}

}
