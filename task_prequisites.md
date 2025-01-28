# Prerequisites and conditions
Shown below you will find the terms and conditions for Subtask 1 of the GermEval2024 GerMS-Detect competition. 

**Note:** This summary is written in my own words. For an exact definition, I recommend visiting the official website: 
[https://ofai.github.io/GermEval2024-GerMS/subtask1.html](https://ofai.github.io/GermEval2024-GerMS/subtask1.html).
 
In Subtask 1, the goal is to predict labels for each text in a dataset derived from the original labels assigned by multiple human annotators.
The human annotators assessed the degree of misogyny/sexism in each text according to the following labels:
- **0 - None:** No sexism/misogyny present  
- **1 - Minimal:** Light sexism/misogyny  
- **2 - Present:** Moderate sexism/misogyny  
- **3 - Strong:** Significant sexism/misogyny  
- **4 - Extreme:** Extreme sexism/misogyny  

While the annotation guidelines define what types of sexism/misogyny should be annotated, they did not provide specific rules for deciding the degree. 
Thus, if an annotator determined that sexism/misogyny was present, the assigned strength was subjective and based on personal judgment.

The labels to be predicted in Subtask 1 reflect various strategies for utilizing multiple annotator labels to derive a final target label:

- **maj:** Predict \`1\` if a majority of annotators assigned a label other than \`0 - None\`; predict \`0\` if a majority assigned \`0 - None\`. If there
is no majority, both \`1\` and \`0\` are considered correct in the evaluation.  
- **bin_one:** Predict \`1\` if at least one annotator assigned a label other than \`0 - None\`; otherwise, predict \`0\`.  
- **bin_all:** Predict \`1\` if all annotators assigned a label other than \`0 - None\`; otherwise, predict \`0\`.  
- **multi_maj:** Predict the majority label if there is one; if no majority exists, any of the assigned labels is considered correct for evaluation.  
- **disagree_bin:** Predict \`1\` if there is disagreement among annotators about \`0 - None\` versus all other labels; otherwise, predict \`0\`.  

### **Data**
For the test phase of Subtask 1, a small dataset is provided, containing:  
- A small labeled dataset with \`id\`, \`text\`, and \`annotations\` (annotator IDs and their assigned labels).  
- A small unlabeled dataset with \`id\`, \`text\`, and \`annotators\` (annotator IDs).  

For the development phase of Subtask 1, participants are provided with:  
- A labeled training dataset containing \`id\`, \`text\`, and \`annotations\` (annotator IDs and their assigned labels).  
- An unlabeled dev set containing \`id\`, \`text\`, and \`annotators\` (annotator IDs).  

For the competition phase of Subtask 1, the following is provided:  
- An unlabeled test set with \`id\`, \`text\`, and \`annotators\` (annotator IDs).  

All five files are in **JSONL** format (one JSON-serialized object per line), where each object is a dictionary with the following fields:  
- **id:** A hash identifying the example.  
- **text:** The text to be classified, which may include arbitrary Unicode and new lines.  
- **annotations (labeled dataset only):** An array of dictionaries containing the following key-value pairs:  
  - **user:** A string in the format \`A003\`, representing an anonymized ID for the annotator who assigned the label.  
  - **label:** The label assigned by the annotator.  

Note that the number of annotations and the specific annotators who provided labels vary between examples.  

- **annotators (unlabeled dataset only):** An array of annotator IDs who labeled the example.  

Data for each phase can be downloaded as soon as the corresponding phase begins.  

---

### **Submission**
The submission must be a **TSV** (tab-separated values) file containing the following columns in any order:  

- **id:** The ID of the example from the unlabeled dataset for which predictions are submitted.  
- **bin_maj:** Prediction of \`0\` or \`1\`.  
- **bin_one:** Prediction of \`0\` or \`1\`.  
- **bin_all:** Prediction of \`0\` or \`1\`.  
- **multi_maj:** Prediction of one of \`0 - None\`, \`1 - Minimal\`, \`2 - Present\`, \`3 - Strong\`, or \`4 - Extreme\`.  
- **disagree_bin:** Prediction of \`1\` or \`0\`.  

You can use multiple models or a single model to generate predictions.  
You may derive the training data for each model in any way from the labeled training data.  
You may choose to use or ignore information about which annotator assigned each label.  

---

### **Evaluation**
The performance of the system across all five predicted labels (\`bin_maj\`, \`bin_one\`, \`bin_all\`, \`multi_maj\`, 
\`disagree_bin\`) will be evaluated using the **F1 macro score** for all classes.  
The final score used to rank submissions will be the unweighted average of the F1 scores across all five labels.  
```
