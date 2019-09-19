import java.io.File;
import java.io.FileNotFoundException;
import java.util.Random;
import java.util.Scanner;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.io.BufferedReader;
import java.io.InputStreamReader;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.io.Text;

public class KMeans {

public static class FeatureRow implements Writable {

	private List<DoubleWritable> features;

	public List<DoubleWritable> getFeatures() {
		return features;
	}

	public void setFeatures(List<DoubleWritable> features) {
		this.features = features;
	}
	
	public void addFeature(FeatureRow ft) {
		for (int i = 0; i < ft.getFeatures().size(); i++) {
			this.features.get(i).set(this.features.get(i).get() + ft.getFeatures().get(i).get());
		}
	}
	
	public FeatureRow(String str) {
	   features = new ArrayList<DoubleWritable>();
	   StringTokenizer itr = new StringTokenizer(str, ",");
	   while (itr.hasMoreTokens()) {
		   DoubleWritable d = new DoubleWritable();
		   try {
		     d.set(Double.parseDouble(itr.nextToken()));
		     features.add(d);
		   } catch(NumberFormatException e) {
		     //not a double
		   }
	   }
   }

   public FeatureRow() {
	   features = new ArrayList<DoubleWritable>();
   }
   
   public FeatureRow(int n) {
	   features = new ArrayList<DoubleWritable>();
	   for (int i = 0; i < n; i++) {
		   features.add(new DoubleWritable(0));
	   }
   }
   
   public void write(DataOutput out) throws IOException {
     (new IntWritable(features.size())).write(out);
	 for (int i = 0; i < features.size(); i++) {
		features.get(i).write(out);
	 }
   }
   
   public void readFields(DataInput in) throws IOException {
     IntWritable counter = new IntWritable();
	 counter.readFields(in); 
	 features = new ArrayList<DoubleWritable>();
	 for (int i = 0; i < counter.get(); i++) {
		DoubleWritable d = new DoubleWritable();
		d.readFields(in);
		features.add(d);
	 	}
   }
   
   @Override
   public boolean equals(Object o) {
	   if (o instanceof FeatureRow) {
		   FeatureRow f = (FeatureRow) o;
		   if (f.getFeatures().size() != this.features.size()) {
			   return false;
		   }
		   for (int i = 0; i < features.size(); i++) {
			   if (!features.get(i).equals(f.getFeatures().get(i))) {
				   return false;
			   }
		   }
		   return true;
	   } else {
		   return false;
	   }
   }
}

	public static class Utils {
		public static double calculateEcluidDist(FeatureRow l1, FeatureRow l2) {
			double dist = 0;
			for (int i = 0; i < l1.getFeatures().size(); i++) {
				double diff = l1.getFeatures().get(i).get() - l2.getFeatures().get(i).get();
				diff *= diff;
				dist += diff;
			}
			return Math.sqrt(dist);
		}
	}

  public static class SemiCentroid implements Writable {
	private IntWritable point_number;
	private FeatureRow row;
	
	public SemiCentroid(int feat) {
		point_number = new IntWritable(0);
		row = new FeatureRow();
		for (int i = 0; i < feat; i++) {
			row.getFeatures().add(new DoubleWritable(0));
		}
	}
	
	public void addFeature(FeatureRow feature) {
		row.addFeature(feature);
		point_number.set(point_number.get() + 1);
	}
	
	@Override
	public void readFields(DataInput arg0) throws IOException {
		point_number.readFields(arg0);
		row.readFields(arg0);
	}

	@Override
	public void write(DataOutput arg0) throws IOException {
		point_number.write(arg0);
		row.write(arg0);
	}

	public IntWritable getPoint_number() {
		return point_number;
	}

	public void setPoint_number(IntWritable point_number) {
		this.point_number = point_number;
	}

	public FeatureRow getRow() {
		return row;
	}

	public void setRow(FeatureRow row) {
		this.row = row;
	}
}


public static class Centroid implements WritableComparable<Centroid> {

	private IntWritable index;
	private FeatureRow row;
	
	public Centroid() {
		index = new IntWritable();
		row = new FeatureRow();
	}
	
	public Centroid(String str) {
		StringTokenizer itr = new StringTokenizer(str);
		index = new IntWritable(Integer.parseInt(itr.nextToken()));
		row = new FeatureRow();
		while (itr.hasMoreTokens()) {
			DoubleWritable d = new DoubleWritable(Double.parseDouble(itr.nextToken()));
			row.getFeatures().add(d);
		}
	}
	
	public Centroid(int index, FeatureRow row, int npoints) {
		this.index = new IntWritable(index);
		this.row = new FeatureRow();
		for (int i = 0; i < row.getFeatures().size(); i++) {
			this.row.getFeatures().add(new DoubleWritable(row.getFeatures().get(i).get() / npoints));
		}
	}
	
	public String toString() {
		StringBuilder s = new StringBuilder();
		s.append(String.valueOf(index.get()));
		for (int i = 0; i < row.getFeatures().size(); i++) {
			s.append(" ");
			s.append(String.valueOf(row.getFeatures().get(i).get()));
		}
		return s.toString();
	}
	
	public IntWritable getIndex() {
		return index;
	}

	public void setIndex(IntWritable index) {
		this.index = index;
	}

	public FeatureRow getRow() {
		return row;
	}

	public void setRow(FeatureRow row) {
		this.row = row;
	}

	@Override
	public void readFields(DataInput arg0) throws IOException {
		index.readFields(arg0);
		row.readFields(arg0);
	}

	@Override
	public void write(DataOutput arg0) throws IOException {
		index.write(arg0);
		row.write(arg0);
	}

	@Override
	public int compareTo(Centroid o) {
		return this.index.compareTo(o.index);
	}
	
	@Override
	public boolean equals(Object o) {
		if (o instanceof Centroid) {
			Centroid c = (Centroid) o;
			return c.getIndex().equals(this.getIndex()) && c.getRow().equals(this.getRow());
		} else {
			return false;
		}
	}
}

public static class CentroidReducer extends Reducer<IntWritable, FeatureRow, Centroid, NullWritable> {

    public void reduce(IntWritable key, Iterable<FeatureRow> values, Context context) throws IOException, InterruptedException {
    	int feat = values.iterator().next().getFeatures().size();
        SemiCentroid sc = new SemiCentroid(feat);
        for (FeatureRow val : values) {
          sc.addFeature(val);
        }
      Centroid c = new Centroid(key.get(), sc.getRow(), sc.getPoint_number().get());
      context.write(c, null);
    }
  }





public static class ClusterMapper extends Mapper<Object, Text, IntWritable, FeatureRow> {

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
    	FeatureRow row = new FeatureRow(value.toString());
    	if (row.getFeatures().size() == 0) {
    		return;
    	}
    	int k = Integer.parseInt(context.getConfiguration().get("k"));
    	int belongs_to = 0;
    	double minDis = Double.MAX_VALUE;
    	for (int i = 0; i < k; i++) {
    		Centroid c = new Centroid(context.getConfiguration().get("c"+String.valueOf(i)));
    		double dist = Utils.calculateEcluidDist(c.getRow(), row);
    		if (dist < minDis) {
    			minDis = dist;
    			belongs_to = i;
    		}
    	}
    	context.write(new IntWritable(belongs_to), row);
	}
  }



public static class FeatureCombiner extends Reducer<IntWritable, FeatureRow, IntWritable, SemiCentroid> {
	
	public void reduce(IntWritable key, Iterable<FeatureRow> values, Context context) throws IOException, InterruptedException {
      int feat = values.iterator().next().getFeatures().size();
      SemiCentroid sc = new SemiCentroid(feat);
      for (FeatureRow val : values) {
        sc.addFeature(val);
      }
      context.write(key, sc);
    }
}



  public static void main(String[] args) throws Exception {
	boolean convergence = false;
	Path inp = new Path(args[0]);
	int k = Integer.parseInt(args[2]);
	int maxIterations = Integer.MAX_VALUE;
	if (args.length > 3) {
		maxIterations = Integer.parseInt(args[3]);
	}
	Centroid[] centroids = new Centroid[k];
	String[] seeds = choose(args[0], k);
	for (int i = 0; i < k; i++) {
		centroids[i] = new Centroid(i, new FeatureRow(seeds[i]), 1);
	}
	int iteration_number = 0;
	while (!convergence && iteration_number < maxIterations) {
	    Configuration conf = new Configuration();
	    conf.set("k", args[2]);
	    for (int i = 0; i < k; i++) {
	    	conf.set("c" + String.valueOf(i), centroids[i].toString());
	    }
	    Job job = Job.getInstance(conf, "K-Means Iteration");
	    job.setJarByClass(KMeans.class);
	    job.setMapperClass(ClusterMapper.class);
	    //job.setCombinerClass(FeatureCombiner.class);
	    job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(FeatureRow.class);
	    job.setReducerClass(CentroidReducer.class);
	    job.setOutputKeyClass(Centroid.class);
	    job.setOutputValueClass(NullWritable.class);
	    FileInputFormat.addInputPath(job, inp);
	    FileOutputFormat.setOutputPath(job, new Path(args[1]));
	    Path dstFilePath = new Path(args[1]);
        try {
            FileSystem fs = dstFilePath.getFileSystem(conf);
            if (fs.exists(dstFilePath))
                fs.delete(dstFilePath, true);
        } catch (IOException e1) {
            e1.printStackTrace();
        }
	    job.waitForCompletion(true);
	    convergence = true;
	    FileSystem fileSystem = FileSystem.get(new Configuration());
		FileStatus[] dir = fileSystem.listStatus(dstFilePath);
		for (FileStatus fs : dir) {
			if (fs.getPath().getName().equals("_SUCCESS")) {
				continue;
			}
		  	BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fileSystem.open(fs.getPath())));
		  	String line = null;
		  	do {
		  		line = bufferedReader.readLine();
		  		if (line != null) {
		  			Centroid c = new Centroid(line);
		  			if (!c.equals(centroids[c.getIndex().get()])) {
		  				convergence = false;
		  			}
		  			centroids[c.getIndex().get()] = c;
		  		}
		  	} while(line != null);
		 }
	    iteration_number++;
	    System.out.println("Iteration " + iteration_number + " finished.");
	}
	System.out.println("Converged in " + iteration_number + " iterations.");
  }
  public static String[] choose(String path, int k) throws FileNotFoundException, IOException
  {
	  String[] result = new String[k];
	  Random rand = new Random();
	  int[] n = new int[k];
	  FileSystem fileSystem = FileSystem.get(new Configuration());
	  FileStatus[] dir = fileSystem.listStatus(new Path(path));
	  for (FileStatus fs : dir) {
	  	BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fileSystem.open(fs.getPath())));
	  	String line = null;
	  	do {
	  		line = bufferedReader.readLine();
	  		if (line != null) {
	  			for (int i = 0; i < k; i++) {
		   	        n[i] = n[i] + 1;
		   	        if(rand.nextInt(n[i]) == 0) {
		   	           result[i] = line;         
		   	        }
		       	 }
	  		}
	  	} while(line != null);
	  }
     return result; 
  }
}
