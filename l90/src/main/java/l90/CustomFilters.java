package l90;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Scanner;

import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.IIndexWord;
import edu.mit.jwi.item.ISenseKey;
import edu.mit.jwi.item.ISynset;
import edu.mit.jwi.item.IWord;
import edu.mit.jwi.item.IWordID;
import edu.mit.jwi.item.POS;
import edu.mit.jwi.item.SenseKey;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.util.CoreMap;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.SparseInstance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;

public class CustomFilters {
	private static HashMap<String, String> synsets;
	private static Properties props = new Properties();
	private static StanfordCoreNLP pipeline;
	private static IDictionary dict;
	
	public static Instances addPosTags(Instances instances, boolean onlyAdjectives) {
		MaxentTagger tagger = new MaxentTagger("/Users/Rashid/Documents/workspace/l90/models/english-left3words-distsim.tagger");
		Instances taggedInstances = new Instances(instances);

		for(int i =0; i<taggedInstances.numInstances(); i++) {
			String text = taggedInstances.instance(i).stringValue(0);
			String tagged = tagger.tagString(text);
			StringBuilder sb = new StringBuilder();
			String[] words = tagged.split(" ");
			for(int j=0; j<words.length; j++) {
				if(onlyAdjectives) {
					if(isAdjAdv(words[j])) sb.append(words[j] + " ");
					else sb.append(words[j].split("_")[0] + " ");
				} else {	
					if(isValid(words[j]))
						sb.append(words[j] + " ");
				}				
			}
			taggedInstances.instance(i).setValue(0, sb.toString());
		}
		return taggedInstances;
	}

	private static boolean isAdjAdv(String word) {
		String[] words = word.split("_");
		String tag = words[words.length-1]; 
		return tag.startsWith("JJ") || tag.startsWith("RB");
	}

	private static boolean isValid(String word) {
		boolean valid = false;
		String[] words = word.split("_");
		if(words.length == 2) {
			String w0 = words[0].replaceAll("[^A-Za-z]", ""); 
			valid =  (w0.length() > 1) ? true : false;
		} else if(words.length > 2) {
			if(words[0].replaceAll("[^A-Za-z]", "").length() > 1 && words[1].replaceAll("[^A-Za-z]", "").length() > 1)
				valid = true;
		} 
		return valid;
	}
	
	private static boolean isValidWord(String word) {
		String w0 = word.replaceAll("[^A-Za-z]", "");
		return w0.length() > 2;
	}

	public static Instances useLexicon(Instances instances) {
		String lexFile = "/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/l90_1/res";
		Instances lexInstances = new Instances(instances);
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		try {
			Scanner scan = new Scanner(new File(lexFile));
			while (scan.hasNextLine()) {
				String[] words = scan.nextLine().split(" ");
				if(!map.containsKey(words[0])) map.put(words[0], Integer.parseInt(words[1]));
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		for(int i=0;i<instances.numInstances(); i++) {
			for(int j=1; j<instances.numAttributes(); j++) {
				Attribute att = instances.instance(i).attribute(j);
				String name = att.name();
				double val = instances.instance(i).value(att);
				if(map.containsKey(name) && val != 0)
					//					att.setWeight((double)map.get(name));
					lexInstances.instance(i).setValue(att, (double)map.get(name));
			}	
		}
		return lexInstances;
	}

	public static Instances useNegation(Instances data, Instances instances) {
		Instances neg = new Instances(instances);

		Attribute att = instances.attribute("not");
		int idx = att.index();

		for(int i=0;i<data.numInstances(); i++) {
			String text = data.instance(i).stringValue(0);
			System.out.println(text);
			String[] words = text.split(" ");

			//iterate over words to find not 
			for(int j=1; j<words.length; j++) {
				String w = words[j].replaceAll("\\s","");
				if(!w.equals("not")) continue;

				Attribute nextWord = instances.attribute(words[j+1]);
				if(nextWord == null) continue;

				double val = instances.instance(i).value(nextWord);
				if(val == 0) continue;
				neg.instance(i).setValue(nextWord, switchPolarity(val));
			}
		}
		return neg;
	}
	
	private static void saveSynSets() throws IOException {
		FileOutputStream fout = new FileOutputStream("synsets.ser");
        ObjectOutputStream oos = new ObjectOutputStream(fout);
        oos.writeObject(synsets);
	}
	
	@SuppressWarnings("unchecked")
	private static void loadSynSets() throws IOException, ClassNotFoundException {
		FileInputStream fin = new FileInputStream("synsets.ser");
        ObjectInputStream oos = new ObjectInputStream(fin);
        synsets = (HashMap<String, String>) oos.readObject();
        int max = 0, min = 99999999;
        //find min and max
        for (String value : synsets.values()) {
            int val = Integer.parseInt(value);
            if(val < min) min = val;
            if(val > max) max = val;
        }
        int diff = max - min;
                
        for (Map.Entry<String, String> entry : synsets.entrySet()) {
            String key = entry.getKey();
            double value = Double.parseDouble(entry.getValue());
            value = (value - min)/diff;
            synsets.put(key, String.valueOf(value));
        }        
	}

	private static double switchPolarity(double polarity) {
		if(polarity > 1)
			polarity = polarity - 2*polarity;
		else if(polarity < 0)
			polarity = -2*polarity + polarity;

		return polarity;
	}

	public static void getGenericFeatureVector() {
		String dataArff = "/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/data/data.arff";
		String vecArff = "/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/data/vec.2000.arff";
		String newShit = "/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/data/vec.csv";
		ArrayList<String> features = new ArrayList<String>();
		ArrayList<String> proper = new ArrayList<String>();

		Instances data = null;
		Instances vec = null;
		try {
			data = new DataSource(dataArff).getDataSet();
			vec = new DataSource(vecArff).getDataSet();
		} catch (Exception e) {
			e.printStackTrace();
		}

		for(int i=0;i<vec.numAttributes(); i++) {
			features.add(vec.attribute(i).name());
		}

		for(int i=0; i<data.numInstances();i++) {
			String text = data.instance(i).stringValue(0);
			String cls = data.instance(i).stringValue(1).equals("NEG") ? "0" : "1"; 
			String res = cls + "," + getVecRepresentation(text, features);
			proper.add(res);
		}

		try {
			FileWriter writer = new FileWriter(newShit);
			for(String str: proper) {
				writer.write(str + "\n");
			}
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 

	}

	private static String getVecRepresentation(String text, ArrayList<String> features) {
		HashSet<String> set = new HashSet<String>();
		String[] words = text.split(" ");
		for(int j=0;j<words.length;j++) {
			set.add(words[j]);
		}

		StringBuilder sb = new StringBuilder();
		for(int i=0;i<features.size();i++) {
			int val = set.contains(features.get(i)) == true ? 1 : 0;
			sb.append(val + ",");
		}

		String res = sb.toString();
		return res.substring(0, res.length() - 1);
	}

	public static Instances UseWordnet(Instances data) throws IOException {
		props.put("annotators", "tokenize,ssplit, pos, lemma");
		pipeline = new StanfordCoreNLP(props);
		synsets = new HashMap<String, String>();
		// construct the URL to the Wordnet dictionary directory
		String wnhome = "/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/data/wordnet";
		String path = wnhome + File.separator + "dict";
		URL url = null;
		try{ url = new URL("file", null, path); } 
		catch(MalformedURLException e){ e.printStackTrace(); }

		// construct the dictionary object and open it
		dict = new Dictionary(url);
		dict.open();
		
		Instances wNetInstances = new Instances(data);
		
		for(int i =0; i<wNetInstances.numInstances(); i++) {
			String text = wNetInstances.instance(i).stringValue(0);
			String tagged = getPOSNet(text);
//			System.out.println(tagged);
			wNetInstances.instance(i).setValue(0,tagged);
		}
		saveSynSets();
		return wNetInstances;
	}
	
	public static void cleanData(Instances instances) {
		for(int i =0; i<instances.numInstances(); i++) {
			String text = instances.instance(i).stringValue(0);
			String clean = text.replaceAll("[^a-zA-Z0-9\\s]","");
			instances.instance(i).setValue(0,clean);
		}
	}
	
	public static Instances getWordNetScores(Instances instances) throws Exception {
		loadSynSets();
		Instances vec = new Instances(instances);

		for(int i=0;i<instances.numInstances(); i++) {
			for(int j=1; j<instances.numAttributes(); j++) {
				Attribute att = instances.instance(i).attribute(j);
				String name = att.name();
				double val = instances.instance(i).value(att);
				if(synsets.containsKey(name) && val != 0) {
//					vec.instance(i).setValue(att, Double.parseDouble(synsets.get(name)));
					String atName = name+"_wn";
					Attribute a = vec.attribute(atName);
					if(a == null) {
						Add filter = new Add();
						filter.setAttributeIndex("last");
				        filter.setAttributeName(atName);
				        filter.setInputFormat(vec);
				        vec = Filter.useFilter(vec, filter);
				        a = vec.attribute(atName);
					}
//					vec.insertAttributeAt(att, vec.numAttributes() + 1);
					vec.instance(i).setValue(a, Double.parseDouble(synsets.get(name)));
				}
			}	
		}
		return vec;
	}

	private static String getPOSNet(String text){ 
		Annotation document = new Annotation(text);
		// run all Annotators on this text
		pipeline.annotate(document); 
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);

		StringBuilder sb = new StringBuilder();
		
		for(CoreMap sentence: sentences) {
			for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
				String word = token.get(TextAnnotation.class);
				word = word.replaceAll("[^a-zA-Z0-9\\s]","");
				if(!isValidWord(word)) continue;
				String lemma = token.get(LemmaAnnotation.class);
				lemma = lemma.replaceAll("[^a-zA-Z0-9\\s]","");
				String pos = token.get(PartOfSpeechAnnotation.class);
				try {
					String synsetId = null;
					POS ps = getSimplePOS(pos);
					if(ps != null) synsetId = getSynsetId(word, ps);
					if(ps == null || synsetId == null) {
						sb.append(word + "_" + pos + " ");
						continue;
					}
					String key = word + "_" + ps.toString(); 
					synsets.put(key, synsetId);
					sb.append(key + " ");
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
		return sb.toString();
	}

	private static POS getSimplePOS(String pos) {
		
		if(pos.startsWith("NN")) return POS.NOUN;
		else if(pos.startsWith("VB")) return POS.VERB;
		else if(pos.startsWith("JJ")) return POS.ADJECTIVE;
		else if(pos.startsWith("RB")) return POS.VERB;
		else return null;	
	}

	private static String getSynsetId(String text, POS pos) throws Exception{
		// look up first sense of the word "dog"
		IIndexWord idxWord = dict.getIndexWord(text, pos);
		if(idxWord == null) return null;
			 	  
		IWordID wordID = idxWord.getWordIDs().get(0);
		IWord word = dict.getWord(wordID);
		ISynset synset = word.getSynset();
		//	    String LexFileName = synset.getLexicalFile().getName();
//		System.out.println(synset);
		return synset.getID().toString().split("-")[1];	
	}
	
	public static Instances addMissingFeatures(Instances data, Instances src) throws Exception {
		Instances res = new Instances(src);
		res.setClassIndex(0);
		Instance temp = new SparseInstance(src.instance(0));
		res.delete();
		
		//add instances
		for(int i=0;i<data.numInstances();i++) {
			Instance in = new SparseInstance(temp);
			in.setDataset(res);
			in.setClassValue(data.instance(i).classValue());
			res.add(in);
			
			for(int j=1;j<in.numAttributes();j++) {
				Attribute at = in.attribute(j);
				Attribute dataAt = data.attribute(at.name());
				
				if(dataAt == null) {
					res.instance(i).setValue(at, 0.0);
				} else {
					res.instance(i).setValue(at, data.instance(i).value(dataAt));
				}
			}
		}
		return res;
	}
	
	private static Add getFilter() throws Exception {
		Add filter = new Add();
		String[] options = new String[2];
		options[0] = "-T";
		options[1] = "NUM";
		filter.setOptions(options);
		return filter;
	}
}
