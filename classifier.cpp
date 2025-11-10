#include <iostream>
#include "csvstream.hpp"
#include <string>
#include <map>
#include <set>
#include <cmath>
#include <cassert>
using namespace std;

class Classifier {
  public:
    //REQUIRES: argc must be 2 (TRAIN ONLY) or 3 (TRAIN + TEST)
    //EFFECTS:  copies argc into a class var for functions to use
    Classifier(int argc) : argc(argc) {
      assert(argc == 2 || argc == 3);
    }

    //REQUIRES: train_file is a good file (valid csvstream)
    //MODIFIES: train_file
    //EFFECTS:  reads each training example, train classifier based on input 
    //          train file, increments total_posts, posts_containing_w, 
    //          posts_with_label_containing_w, posts_with_label_c, inserts words
    //          into vocab and tags into labels, prints training output
    void train(csvstream &train_file) {
      assert(train_file);

      map<string, string> post;

      if (argc == 2) {
        cout << "training data:\n";
      }

      while (train_file >> post) {
        total_posts++;

        for (string i : unique_words(post["content"])) {
          vocab.insert(i);
          posts_containing_w[i]++;
          posts_with_label_containing_w[{post["tag"], i}]++;
        }

        posts_with_label_C[post["tag"]]++;
        labels.insert(post["tag"]);

        if (argc == 2) {
          cout << "  label = " << post["tag"] 
               << ", content = " << post["content"] << endl;
        }
      }
      
      cout << "trained on " << total_posts << " examples\n";

      if (argc == 2) {
        cout << "vocabulary size = " << vocab.size() << "\n\n";
        print_classifier_info();
        cout << "\n"; // extra \n for train only
      }
    }

    //REQUIRES: classifier has been trained (total_posts > 0)
    //EFFECTS:  print classifier info: summary stats, and classifier parameters
    //          for each label
    void print_classifier_info() {
      assert(total_posts > 0); // ASK PROFESSOR
      cout << "classes:\n";

      for (string label : labels) {
        cout << "  " << label << ", " << posts_with_label_C[label]
             << " examples, log-prior = " << calc_prior(label) << endl;
      }

      cout << "classifier parameters:\n";

      for (string label : labels) {
        for (string w : vocab) {
          if (posts_with_label_containing_w[{label, w}] > 0){
            map<string, string> temp;
            temp["content"] = w;
            cout << "  " << label << ":" << w
                << ", count = " << posts_with_label_containing_w[{label, w}]
                << ", log-likelihood = " << calc_likelihood(temp, label)
                << endl;
          }
        }
      }
    }

    //REQUIRES: test_file is a good file
    //MODIFIES: test_file
    //EFFECTS:  test classifier based on input test file
    void test(csvstream &test_file) {
      assert(test_file);

      cout << "\n";
      cout << "test data:\n";

      map<string, string> post;
      int correct = 0, total_tested = 0;

      while (test_file >> post) {
        pair<string, double> res = predict(post);

        if (res.first == post["tag"]) {
          correct++;
        }

        cout << "  correct = " << post["tag"] << ", predicted = "<< res.first
             << ", log-probability score = " << res.second << endl;
        cout << "  content = " << post["content"] << "\n\n";

        total_tested++;
      }

      cout << "performance: " << correct << " / " << total_tested
           << " posts predicted correctly\n";
    }

    //REQUIRES: Classifier has been trained
    //EFFECTS:  given a post, predicts label with probability
    // should take in a post (row in csv)
    pair<string, double> predict(map<string, string> &post) {
      string label_max = "";
      double prob_max;

      // loop through all possible labels
      for (string label : labels) {
        // calculate log-probability score
        double prob = calc_prior(label) + calc_likelihood(post, label);

        if (label_max == "") {
          prob_max = prob;
          label_max = label;
        } else if (prob > prob_max) {
          prob_max = prob;
          label_max = label;
        }
      }

      return {label_max, prob_max};
    }

    //REQUIRES: str is a whitespace-separated string (could be empty)
    //EFFECTS: splits a string into unique words
    set<string> unique_words(const string &str) {
      istringstream source(str);
      set<string> words;
      string word;
      while (source >> word) {
        words.insert(word);
      }
      return words;
    }

  private:
    int argc;

    // keep track of the classifier parameters learned from the training data

    // total number of posts in the entire training set
    int total_posts;
    // number of unique words in the entire training set
    // labels seen in training set
    set<string> vocab, labels;
    // number of posts in the entire training set that contain each word w
    // number of posts with each label C
    map<string, int> posts_containing_w, posts_with_label_C;
    // number of posts with label C that contain w
    map<pair<string, string>, int> posts_with_label_containing_w;

    //REQUIRES: total_points > 0
    //EFFECTS:  calculate log-prior probability
    double calc_prior(string label) {
      assert(total_posts > 0);
      return log(posts_with_label_C[label] / static_cast<double>(total_posts));
    }

    //EFFECTS:  calculate log-likelihood
    double calc_likelihood(map<string, string> &post, string &label) {
      // create a set of unique words in post
      set<string> words = unique_words(post["content"]);
      double likelihood = 0;
      for (string i : words) {
        if (posts_with_label_containing_w[{label, i}] == 0 &&
            posts_containing_w[i] == 0) {
          likelihood += log(1.0 / total_posts);
        } else if (posts_with_label_containing_w[{label, i}] == 0) {
          likelihood +=
            log(posts_containing_w[i] / static_cast<double>(total_posts));
        } else {
          likelihood += log(posts_with_label_containing_w[{label, i}] /
                            static_cast<double>(posts_with_label_C[label]));
        }
      }
      return likelihood;
    }
};

int main(int argc, char *argv[]) {
  cout.precision(3);

  //Check if we have the right amount of arguments
  if ((argc != 2) && (argc != 3)) {
    cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
    return 1;
  }

  //Initialize Classifier
  Classifier classifier(argc);

  // open TRAIN_FILE
  try {
    csvstream train_file(argv[1]);
    //Train model
    classifier.train(train_file);
  } catch (const csvstream_exception &e) {
    //Error opening file
    cout << "Error opening file: " << argv[1] << endl;
    return 1;
  }

  //Train + Test
  if (argc == 3) {
    // open TEST_FILE
    try {
      csvstream test_file(argv[2]);
      //Test the file
      classifier.test(test_file);
    } catch (const csvstream_exception &e) {
      //Error opening file
      cout << "Error opening file: " << argv[2] << endl;
      return 1;
    }
  }
}