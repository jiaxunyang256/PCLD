# CPCLDetector

**CPCLDetector: Knowledge Enhancement and Alignment Selection for Chinese Patronizing and Condescending Language Detection**

---

## Overview

CPCL refers to implicitly toxic speech targeting vulnerable groups on Chinese video platforms (e.g., Douyin, Bilibili). Unlike explicit hate speech, it conveys "superiority" through hypocritical/condescending attitudes, making detection challenging.

**Core components**

* Alignment Selection Module: Unifies spatiotemporal dimensions of multi-source features (video, facial expression, audio, text) and reduces cross-modal distribution differences.
* Alignment Selection Module: Unifies spatiotemporal dimensions of multi-source features (video, facial expression, audio, text) and reduces cross-modal distribution differences.


**Key contributions**

* Constructed PCLMMPLUS, a expanded multi-modal dataset with 831 video samples and 103k user comments, filling the comment modality blank.
* Proposed CPCLDetector, which outperforms state-of-the-art (SOTA) models on both the original PCLMM dataset and the new PCLMMPLUS dataset.
* Verified the synergistic value of alignment selection and knowledge-enhanced comment modules via ablation experiments, with significant improvements in recall (critical for capturing implicit CPCL).

---

## Datasets

**Dataset Comparison**

<table width="560" border="0" cellpadding="0" cellspacing="0" style='width:336.00pt;border-collapse:collapse;table-layout:fixed;'>
   <col width="80" span="7" style='width:48.00pt;'/>
   <tr height="125.00" style='height:75.00pt;'>
    <td class="xl65" height="125.00" width="80" style='height:75.00pt;width:48.00pt;' x:str>Statistics</td>
    <td class="xl66" width="240" colspan="3" style='width:144.00pt;border-right:none;border-bottom:none;' x:str>PCLMM (Prior)</td>
    <td class="xl66" width="240" colspan="3" style='width:144.00pt;border-right:none;border-bottom:none;' x:str>PCLMMPLUS (Ours)</td>
   </tr>
   <tr height="50" style='height:30.00pt;'>
    <td class="xl67" height="50" style='height:30.00pt;'></td>
    <td class="xl68" x:str>Non-PCL</td>
    <td class="xl68" x:str>PCL</td>
    <td class="xl68" x:str>Total</td>
    <td class="xl68" x:str>Non-PCL</td>
    <td class="xl68" x:str>PCL</td>
    <td class="xl68" x:str>Total</td>
   </tr>
   <tr height="75" style='height:45.00pt;'>
    <td class="xl67" height="75" style='height:45.00pt;' x:str>Videos Total Num</td>
    <td class="xl68" x:num>519</td>
    <td class="xl68" x:num>196</td>
    <td class="xl68" x:num>715</td>
    <td class="xl68" x:num>519</td>
    <td class="xl68" x:num>312</td>
    <td class="xl68" x:num>831</td>
   </tr>
   <tr height="100" style='height:60.00pt;'>
    <td class="xl67" height="100" style='height:60.00pt;' x:str>Videos Total Length (hrs)</td>
    <td class="xl68" x:num>15.1</td>
    <td class="xl68" x:num>6.5</td>
    <td class="xl68" x:num>21.6</td>
    <td class="xl68" x:num>15.1</td>
    <td class="xl68" x:num>11.9</td>
    <td class="xl68" x:num>27</td>
   </tr>
   <tr height="100" style='height:60.00pt;'>
    <td class="xl67" height="100" style='height:60.00pt;' x:str>Comments Total Num</td>
    <td class="xl68" x:str>-</td>
    <td class="xl68" x:str>-</td>
    <td class="xl68" x:str>-</td>
    <td class="xl68" x:str>77K</td>
    <td class="xl68" x:str>26K</td>
    <td class="xl68" x:str>103K</td>
   </tr>
   <tr height="125.00" style='height:75.00pt;'>
    <td class="xl67" height="125.00" style='height:75.00pt;' x:str>Comments Total Length (chars)</td>
    <td class="xl68" x:str>-</td>
    <td class="xl68" x:str>-</td>
    <td class="xl68" x:str>-</td>
    <td class="xl68" x:str>2935K</td>
    <td class="xl68" x:str>992K</td>
    <td class="xl68" x:str>3927K</td>
   </tr>
   <![if supportMisalignedColumns]>
    <tr width="0" style='display:none;'/>
   <![endif]>
  </table>

**Dataset Comparison**

* PCLMM Dataset:

    **1.** The first multi-modal CPCL dataset for Chinese videos, targeting 6 vulnerable groups from Bilibili.
    
    **2.** Contains 715 annotated samples (196 PCL, 519 Non-PCL), but lacks comment data and has imbalanced PCL samples.

* PCLMMPLUS Dataset:

    **1.** Inherits PCLMM’s core design, adds 116 new PCL video samples (total 312 PCL samples) with 5.4 additional hours of content.
    
    **2.** Includes 103k cleaned first-level user comments (via platform-compliant interfaces).

    **3** Provides more comprehensive context for CPCL detection and alleviates sample imbalance.
---

## Model Architecture
**Alignment Selection Module**

* Unifies feature dimensions across modalities and aligns them to text (as the anchor) for efficient fusion. 

**Knowledge-Enhanced Comment Content Module**

* Knowledge-Enhanced Sentiment Analysis
* Comment Information Processing
---

## Experimental Results

### Performance on PCLMM Dataset (SOTA Comparison)

  <table width="650" border="0" cellpadding="0" cellspacing="0" style='width:390.00pt;border-collapse:collapse;table-layout:fixed;'>
   <col width="208" style='mso-width-source:userset;mso-width-alt:6085;'/>
   <col width="108.00" style='mso-width-source:userset;mso-width-alt:3159;'/>
   <col width="125.00" style='mso-width-source:userset;mso-width-alt:3657;'/>
   <col width="102.00" style='mso-width-source:userset;mso-width-alt:2984;'/>
   <col width="107" style='mso-width-source:userset;mso-width-alt:3130;'/>
   <tr height="25" style='height:15.00pt;'>
    <td class="xl65" height="25" width="208" style='height:15.00pt;width:124.80pt;' x:str>Model</td>
    <td class="xl66" width="108.00" style='width:64.80pt;' x:str>Accuracy</td>
    <td class="xl66" width="125.00" style='width:75.00pt;' x:str>F1 (macro)</td>
    <td class="xl66" width="102.00" style='width:61.20pt;' x:str>Recall</td>
    <td class="xl66" width="107" style='width:64.20pt;' x:str>Precision</td>
   </tr>
   <tr height="25" style='height:15.00pt;'>
    <td class="xl67" height="25" style='height:15.00pt;' x:str>BERT-PCL</td>
    <td class="xl66" x:num>0.7972</td>
    <td class="xl66" x:num>0.7113</td>
    <td class="xl66" x:num>0.5294</td>
    <td class="xl66" x:num>0.5806</td>
   </tr>
   <tr height="25" style='height:15.00pt;'>
    <td class="xl67" height="25" style='height:15.00pt;' x:str>GPT4</td>
    <td class="xl66" x:num>0.8252</td>
    <td class="xl66" x:num>0.7455</td>
    <td class="xl66" x:num>0.5588</td>
    <td class="xl66" x:num>0.6552</td>
   </tr>
   <tr height="25" style='height:15.00pt;'>
    <td class="xl67" height="25" style='height:15.00pt;' x:str>VideoMAE</td>
    <td class="xl66" x:num>0.7778</td>
    <td class="xl66" x:num>0.709</td>
    <td class="xl66" x:num>0.525</td>
    <td class="xl66" x:num>0.6176</td>
   </tr>
   <tr height="25" style='height:15.00pt;'>
    <td class="xl67" height="25" style='height:15.00pt;' x:str>MultiPCL (SOTA) [9]</td>
    <td class="xl66" x:num>0.8309</td>
    <td class="xl66" x:num>0.7978</td>
    <td class="xl66" x:num>0.7632</td>
    <td class="xl66" x:num>0.6744</td>
   </tr>
   <tr height="26" style='height:15.60pt;'>
    <td class="xl68" height="26" style='height:15.60pt;' x:str>CPCLDetector (Ours)</td>
    <td class="xl69" x:num>0.8382</td>
    <td class="xl69" x:num>0.813</td>
    <td class="xl69" x:num>0.8878</td>
    <td class="xl69" x:num>0.7317</td>
   </tr>
   <![if supportMisalignedColumns]>
    <tr width="0" style='display:none;'>
     <td width="208" style='width:125;'></td>
     <td width="108" style='width:65;'></td>
     <td width="125" style='width:75;'></td>
     <td width="102" style='width:61;'></td>
     <td width="107" style='width:64;'></td>
    </tr>
   <![endif]>
  </table>

> *Improvements vs. MultiPCL: +0.73% Accuracy, +1.52% F1, +12.46% Recall, +5.73% Precision.*

> *Statistical Significance: Paired t-tests show p<0.05 for Accuracy and F1.*
---

### Performance on PCLMMPLUS Dataset

Benefiting from expanded samples and comment data, CPCLDetector maintains strong performance:

* Accuracy: 0.8603 (+2.21% vs. PCLMM)
* F1 (macro): 0.8305
* Recall: 0.8421 (high retention)
* Precision: 0.6667 (acceptable, with effective noise filtering)
* Statistical Significance: One-way ANOVA shows p<0.05 for Accuracy and F1.

---

### Ablation Experiments (PCLMMPLUS Dataset)

<table width="526" border="0" cellpadding="0" cellspacing="0" style='width:315.60pt;border-collapse:collapse;table-layout:fixed;'>
   <col width="256" style='mso-width-source:userset;mso-width-alt:7489;'/>
   <col width="108.00" style='mso-width-source:userset;mso-width-alt:3159;'/>
   <col width="162" style='mso-width-source:userset;mso-width-alt:4739;'/>
   <tr height="50" style='height:30.00pt;'>
    <td class="xl65" height="50" width="256" style='height:30.00pt;width:153.60pt;' x:str>Model Variant</td>
    <td class="xl66" width="108.00" style='width:64.80pt;' x:str>Accuracy</td>
    <td class="xl66" width="162" style='width:97.20pt;' x:str>Accuracy Drop (vs. Full Model)</td>
   </tr>
   <tr height="25" style='height:15.00pt;'>
    <td class="xl67" height="25" style='height:15.00pt;' x:str>Full Model (CPCLDetector)</td>
    <td class="xl66" x:num>0.8603</td>
    <td class="xl66" x:str>–</td>
   </tr>
   <tr height="50" style='height:30.00pt;'>
    <td class="xl67" height="50" style='height:30.00pt;' x:str>Full Model – Knowledge-Enhanced Sentiment Module</td>
    <td class="xl66" x:num>0.8469</td>
    <td class="xl68" x:num="1.34e-002">1.34%</td>
   </tr>
   <tr height="50" style='height:30.00pt;'>
    <td class="xl67" height="50" style='height:30.00pt;' x:str>Full Model – Comment Processing Module</td>
    <td class="xl66" x:num>0.8451</td>
    <td class="xl68" x:num="1.52e-002">1.52%</td>
   </tr>
   <tr height="25" style='height:15.00pt;'>
    <td class="xl67" height="25" style='height:15.00pt;' x:str>Full Model – Both Modules</td>
    <td class="xl66" x:num>0.8361</td>
    <td class="xl68" x:num="2.4199999999999999e-002">2.42%</td>
   </tr>
   <![if supportMisalignedColumns]>
    <tr width="0" style='display:none;'>
     <td width="256" style='width:154;'></td>
     <td width="108" style='width:65;'></td>
     <td width="162" style='width:97;'></td>
    </tr>
   <![endif]>
  </table>

> *Statistical Significance: T-tests show p<0.05 for all ablation variants.*
---




```

---
