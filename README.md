# Automated-Essay-Scoring-Web-App
The "Automated Essay Scoring Web App" leverages natural language processing (NLP) algorithms to evaluate the content, coherence, and grammar of an essay and assign it a score on a predetermined scale. The repository includes the backend server code, frontend user interface code, and various scripts and dependencies required to run the application.
Users can upload their essays to the web app and receive instant feedback on their writing skills. The app provides detailed scoring metrics and feedback on various aspects of the essay, including its organization, argument structure, vocabulary usage, and more. The app is highly customizable and can be tailored to meet the specific needs of different users, such as teachers, students, or professional writers.

## Motivation
The motivation for the "Automated Essay Scoring Web App" project is to provide an efficient and objective way of assessing written assignments in educational institutions. Traditional methods of grading essays can be time-consuming and subjective, leading to inconsistencies and biases in the evaluation process. With an automated essay scoring system, the evaluation process can be streamlined and standardized, providing more accurate and reliable grading on students' writing skills.
"Automated Essay Scoring Web App" project aims to make essay grading more accessible and available to students and teachers worldwide. By offering a free, open-source solution, the project helps level the playing field and provides opportunities for learners from all backgrounds to benefit from modern technology.

## Dataset
The dataset we are using is ‘The Hewlett Foundation: Automated Essay Scoring Dataset’ by ASAP. You can find in the Dataset folder.

## Architecture Diagram
![Automated-Essay-Scoring-Web-App-Architecture](https://github.com/KratoSkills/Automated-Essay-Scoring-Web-App/assets/56100874/0e831906-b482-4523-a506-1b0859fe68df)

## Proposed Model
Initially, we compile a list of words from each sentence and essay and input this data into the Word2Vec model. This allows us to derive numerical vector values for each word, thereby making sense of the available vocabulary. Next, we utilize the Word2Vec model as an Embedding Layer within a neural network to generate features for the essays. Specifically, we implement two LSTM layers in our model. The first layer takes in all the features from the Embedding Layer (Word2Vec) as input and outputs 300 features to the second LSTM layer. Then, the second layer accepts 300 features as input and produces 64 features as output. We incorporate a Dropout layer with a value of 0.5 and conclude with a fully connected Dense Layer that outputs the score of the essay as 1. To train this model, we compiled it using the Mean Squared Error loss function and Root Mean Square optimizer, and trained it over 150 epochs with a batch size of 64.

However, we first divide the resulting model into four distinct modules for ease of understanding and optimization, which are as follows:

1. Data Preprocessing
Initially, we conducted standard preprocessing techniques such as filling null values and carefully selecting relevant features from the complete dataset. We then plotted a graph to determine the data's skewness and employed normalisation methods to reduce this skewness. Additionally, we cleaned the essays to facilitate our training process for improved accuracy. We removed unnecessary symbols, stop words, and punctuation from our essays. To further enhance our accuracy, we also planned to incorporate additional features, such as sentence count, word count, character count, and average word length. Furthermore, we implemented techniques like parts of speech tagging to obtain noun, verb, adjective, and adverb counts, and compared essay text with a corpus to determine total misspellings. In the next section, we discuss various machine learning algorithms that we applied to this data.

  The processed dataset can be located in the file named "Processed_data.csv".

**2. Machine Learning**
Before applying machine learning algorithms to our dataset, we need to take an additional step. Machine learning algorithms only work with numerical data, so we must first convert the essays in our dataset into a numeric form. We accomplish this by using a CountVectorizer, which tokenizes a collection of text documents and returns an encoded vector with a length equal to the entire vocabulary, along with an integer count for each word that appears in the document. After this crucial step, our data is ready for predictive modeling.
Initially, we applied machine learning algorithms such as linear regression, SVR, and Random Forest to our dataset without incorporating the additional features mentioned in the preprocessing section above. Unfortunately, our results were unsatisfactory, as the mean squared error was quite high for all three algorithms. Afterwards, we included the extra features, applied the CountVectorizer once more to the modified dataset, and once again utilized the same aforementioned algorithms. There was a significant improvement in the performance of all three algorithms, particularly with Random Forest, which experienced a drastic reduction in mean squared error.

For implementation details of this module, please refer to the "essayScoring.ipynb" file located in the repository.

**3. Applying Neural Networks**
Preprocessing steps for neural networks are distinct from those for machine learning algorithms. In our particular case, we feed our training data into the Word2Vec Embedding Layer. Word2Vec represents a shallow, two-layer neural network that is trained to reconstruct the linguistic context of words. It takes in a large corpus of words and produces a vector space, generally consisting of several hundred dimensions. Each unique word in the corpus is assigned a corresponding vector within this space. Word vectors are arranged in the vector space so that words with shared contexts in the corpus are located close to one another. Word2Vec is an especially computationally-efficient predictive model for learning word embeddings from raw text. The features derived from Word2Vec are then fed into LSTM, which can recognize the importance of certain data in a sequence and decide whether to keep or discard it. Ultimately, this aids in calculating scores from essays. Finally, the Dense layer with output 1 predicts the score of each essay.

For further implementation details on this module, please refer to the "essayScoringNeuralNetwork.ipynb" file within the repository.
![essayScoringNeuralNetworkModel](https://github.com/KratoSkills/Automated-Essay-Scoring-Web-App/assets/56100874/6ec0f091-ac45-457d-8494-4fc0d3db7d9f)

**4. Creation of Web Application**
After successfully training our model, our next objective was to make the project accessible to users by creating a web application. To accomplish this task, we employed the Flask framework to deploy our model. Flask is a well-known Python web framework that serves as a third-party library used for developing web applications. By utilizing Flask, we were able to create an API that receives essay details through a graphical user interface (GUI) and calculates the predicted score value based on our trained model. Results are displayed via a POST request, whereby JSON inputs are received, the trained model is utilized to generate a prediction, and the prediction is returned in JSON format through the API endpoint.

The core component of the web application can be found in the "webApp" folder of the repository. Additionally, screenshots of the web page can be accessed through the following links:
  - Step 1:
![Step 1](https://github.com/KratoSkills/Automated-Essay-Scoring-Web-App/assets/56100874/e1ad72ad-b58e-424f-8b48-a715e95d646e)
  - Step 2:
![Step 2](https://github.com/KratoSkills/Automated-Essay-Scoring-Web-App/assets/56100874/7de8ae78-9b77-4569-83cc-acec716ef157)
  - Step 3:
![Step 3](https://github.com/KratoSkills/Automated-Essay-Scoring-Web-App/assets/56100874/1954920f-bff9-4a4d-b254-4850d8bee135)

## Conclusion
We developed a deep neural network model that can effectively consider both local and contextual information in essay scoring. The model generates unique word embeddings that correspond to specific scores, which are then utilized by a recurrent neural network to construct essay representations. Our approach outperformed other state-of-the-art systems, and we introduced an innovative technique for exploring the network's internal scoring criteria, demonstrating that our models are interpretable and capable of providing useful feedback to authors.

Our most successful neural network model used a 300-dimensional LSTM as initialization to the embedding layer, and we believe that conducting a more extensive hyperparameter search with our LSTM-based models could yield even better results. We have several ideas for future research, including experimenting with ensemble mode models.
