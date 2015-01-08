package l90;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.documentiterator.FileDocumentIterator;
import org.deeplearning4j.text.inputsanitation.InputHomogenization;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;

/**
 * Created by agibsonccc on 9/20/14.
 */
public class Word2VecExample {
    private SentenceIterator iter;
    private TokenizerFactory tokenizer;
    private Word2Vec vec;
    private File training;

    public final static String VEC_PATH = "vec2.ser";
    public final static String CACHE_SER = "cache.ser";

    public Word2VecExample(File training, String path) throws Exception {
    	this.training = training;
        this.iter = new LineSentenceIterator(new File(path));
        tokenizer =  new DefaultTokenizerFactory();
    }

    public static void main(String[] args) throws Exception {
            File f = new File("/Users/Rashid/dev/Cambridge/L90_Overview_of_NLP/data/train_s");
            new Word2VecExample(f, f.getAbsolutePath()).train();
    }

    public void train() throws Exception {
    	SentenceIterator iter = new FileSentenceIterator(training);
//    	TokenizerFactory t = new UimaTokenizerFactory();
    	
    	vec = new Word2Vec.Builder().windowSize(5).layerSize(300).iterate(iter).tokenizerFactory(tokenizer).build();
    	
    	
//        vec = new Word2Vec.Builder().minWordFrequency(5).vocabCache(cache)
//                .windowSize(5)
//                .layerSize(100).iterate(iter).tokenizerFactory(tokenizer)
//                .build();
//        vec.setCache(cache);
        vec.fit();

        SerializationUtils.saveObject(vec, new File(VEC_PATH));
//        SerializationUtils.saveObject(cache,new File(CACHE_SER));           
    }
}