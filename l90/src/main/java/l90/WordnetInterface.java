package l90;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;

import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.IIndexWord;
import edu.mit.jwi.item.ISenseKey;
import edu.mit.jwi.item.ISynset;
import edu.mit.jwi.item.IWord;
import edu.mit.jwi.item.IWordID;
import edu.mit.jwi.item.POS;
import edu.mit.jwi.item.SenseKey;

public class WordnetInterface {
//	private static void 
	
	
	public static void runExample() throws IOException{
		   
		     // construct the URL to the Wordnet dictionary directory
		     String wnhome = "/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/data/wordnet";
		     String path = wnhome + File.separator + "dict";
		     URL url = null;
		     try{ url = new URL("file", null, path); } 
		     catch(MalformedURLException e){ e.printStackTrace(); }
		     if(url == null) return;
		    
		    // construct the dictionary object and open it
		    IDictionary dict = new Dictionary(url);
		    dict.open();
		
		    // look up first sense of the word "dog"
		    IIndexWord idxWord = dict.getIndexWord("hellish", POS.ADJECTIVE);
		    IWordID wordID = idxWord.getWordIDs().get(0);
		    IWord word = dict.getWord(wordID);
		    ISynset synset = word.getSynset();
//		    String LexFileName = synset.getLexicalFile().getName();
		    System.out.println(synset);
		    System.out.println(synset.getID().toString().split("-")[1]);

		    System.out.println(word);
		    System.out.println("Id = " + wordID);
		    System.out.println("Lemma = " + word.getLemma());
		    System.out.println("Gloss = " + word.getSynset().getGloss());
		
	}

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		runExample();
	}

}
