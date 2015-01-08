package l90;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.*;

import org.apache.commons.io.FileUtils;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.PTBTokenizer;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.IteratedLovinsStemmer;
import weka.core.stemmers.LovinsStemmer;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;


public class SentimentAnalysis {
		private static String trainArff = "/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/data/train.arff";
		private static String testArff = "/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/data/test.arff";
		private static String dataArff = "/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/data/data.arff";
		private static String posArff = "/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/data/pos.arff";
		private static String vecArff = "/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/data/vec.arff";
		private static String lexArff = "/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/data/vec_lex.arff";
		private static String wnetArff = "/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/data/pos_wnet.arff";
		private static String snetArff = "/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/data/pos_snet.arff";
		
		private static Instances trainSet;
		private static Instances testSet;
		private static FilteredClassifier classifier;
		private static String naiveModel = "NaiveBayes.model";
		private static String svmModel = "SVM.model";		
	public static void main(String[] args) throws Exception {
//		trainClassifier();
		usePretrainedModel();
	}
	
	private static void trainClassifier() {	
		try {			
//			SentiWordBuilder sw = new SentiWordBuilder("/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/data/SW_3.0.txt");
//			StringToWordVector filter = getFilter();
//			FilteredClassifier classifier = new FilteredClassifier();
//			Instances data = new DataSource(dataArff).getDataSet();
//			CustomFilters.cleanData(data);
//			data.setClassIndex(1);
//			filter.setInputFormat(data);
			
//			Instances posTagged = sw.getPosTagged(data); 
//			Instances vec = Filter.useFilter(data,filter);
//			Instances snet = sw.useSentiWordVals(vec, true);
			Instances snet = new DataSource(snetArff).getDataSet();
			snet.setClassIndex(0);
						
//			Instances neg = CustomFilters.useNegation(data, lex);
			
//			ArffSaver saver = new ArffSaver();
//			saver.setInstances(snet);
//			saver.setFile(new File(snetArff)); 
//			saver.writeBatch();	

			Classifier cls = new SMO();

			Evaluation eval = new Evaluation(snet);
			System.out.println("Training model");			
			eval.crossValidateModel(cls, snet, 5, new Random(1));
			System.out.println(eval.toSummaryString());
			
			cls.buildClassifier(snet);			
			saveModel(cls, "SMO");
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private static void saveModel(Classifier cls, String name) throws IOException {
		ObjectOutputStream oos = null;
        oos = new ObjectOutputStream(new FileOutputStream(name + ".model"));	    
	    oos.writeObject(cls);
	    oos.flush();
	    oos.close();
	}
	
	private static Classifier loadModel(String name) throws Exception {
	    Classifier classifier;
	    FileInputStream fis = new FileInputStream(name + ".model");
	    ObjectInputStream ois = new ObjectInputStream(fis);
	    classifier = (Classifier) ois.readObject();
	    ois.close();
	    return classifier;
	}

	private static void usePretrainedModel() throws Exception {
		SentiWordBuilder sw = new SentiWordBuilder("/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/data/SW_3.0.txt");
		StringToWordVector filter = getFilter();
		Instances data = new DataSource(testArff).getDataSet();
		CustomFilters.cleanData(data);
		data.setClassIndex(1);
		filter.setInputFormat(data);
		
		//add pos tags
		Instances pos = sw.getPosTagged(data);
		Instances vec = Filter.useFilter(pos,filter);
		
		//add missing features
		vec = CustomFilters.addMissingFeatures(vec, new DataSource(snetArff).getDataSet());
		Instances snet = sw.useSentiWordVals(vec, true);
		
		SMO cls = (SMO) loadModel("SMO");
		for(int i =0;i<snet.numInstances();i++) {
			Instance t = snet.instance(i);
			double out = cls.classifyInstance(t);
			
			if(out != snet.instance(i).classValue())
			System.out.println(data.instance(i).stringValue(0));
		}
	}
	
	private static void useLexicon() {
		try {
			Instances data = new DataSource(dataArff).getDataSet();
			Instances lex = new DataSource(lexArff).getDataSet();
			lex.setClassIndex(0);
			Instances neg = CustomFilters.useNegation(data, lex);
			
			Evaluation eval = new Evaluation(neg);
			eval.crossValidateModel(new SMO(), neg, 3, new Random(1)); 
//			
			
			System.out.println(eval.toSummaryString());	
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
	}
	
	private static StringToWordVector getFilter() {
		StringToWordVector filter = new StringToWordVector();
		NGramTokenizer tokenizer = new NGramTokenizer();
		LovinsStemmer stemmer = new LovinsStemmer();
		
		tokenizer.setNGramMaxSize(3);
		tokenizer.setNGramMinSize(1);
		
		filter.setWordsToKeep(15000);
//		filter.setOutputWordCounts(true);
//		filter.setMinTermFreq(5);
//		filter.setTFTransform(true);
//		filter.setIDFTransform(true);
		filter.setLowerCaseTokens(true);
		filter.setTokenizer(tokenizer);
//		filter.setStemmer(stemmer);
//		filter.setUseStoplist(true);
		return filter;
	}
}
