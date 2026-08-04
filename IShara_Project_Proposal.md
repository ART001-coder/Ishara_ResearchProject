## PROJECT PROPOSAL

# IShara

## An Indian Sign Language to Text Translation System with Language-Model-Assisted Sentence Reconstruction

Prepared for: [Name of Reviewing Authority / Department]

Prepared by: [Team Member Names] Institution: [Institution / Organisation Name]

Date: [Presentation Date]


## 1. Executive Summary

Communication between users of Indian Sign Language (ISL) and the wider hearing population remains difficult in everyday settings such as hospitals, banks, and government offices, largely because no widely available tool converts live ISL signing into readable text. This project develops a working prototype that recognises a sequence of individual ISL signs from webcam video and reconstructs them into a grammatically correct sentence using a language model.

The prototype is scoped to a focused vocabulary of 100–150 words centred on a hospital and reception communication scenario. It is built primarily on the publicly available INCLUDE dataset from IIT Madras, supplemented with data recorded by the team for words outside that dataset. The complete pipeline — from a signer performing a sequence of words in front of a webcam to a finished, readable sentence — is demonstrated live at the end of a two-week development cycle.

## 2. Problem Statement

Existing ISL research has largely focused on recognising individual signs in isolation. In practice, a sequence of individual signs does not read as a natural sentence: ISL word order differs from English, articles and tense markers are often not signed explicitly, and recognition errors on individual signs are common. A usable system therefore needs a second stage that takes the raw, imperfect output of a sign recognition model and reconstructs it into text a hearing person can read naturally. This project treats that second stage as a first-class part of the system rather than an afterthought.

## 3. Objectives

- 1. Build a sign-recognition model that correctly classifies a defined vocabulary of ISL words from webcam video.

- 2. Design a language-model correction layer that converts a raw sequence of recognised words into a fluent, grammatically sound sentence.

- 3. Demonstrate the complete pipeline live, from signing to displayed sentence, within the two-week timeline.

- 4. Establish an architecture that can be extended later to a larger vocabulary, continuous signing, and additional platforms.

## 4. Project Scope

To keep the project achievable within two weeks with a five-person team, the following boundaries were agreed at the outset.

| In Scope | Out of Scope |
| --- | --- |
| Isolated-sign recognition for a fixed vocabulary of 100– 150 words | Mobile app deployment (Android/iOS) |
| Language-model-based sentence reconstruction from recognised word sequences | Edge / kiosk hardware deployment |
| Real-time webcam demo on a laptop | Regional dialect handling |
| Confidence-based handling of uncertain recognitions | Bidirectional translation (text to ISL) |
| Evaluation on signers outside the training set | Fingerspelling recognition |
| Hospital / reception communication scenario as the demo context | Real-time continuous fluent signing |


## 5. Related Work and Grounding

The project builds on an existing body of ISL research rather than starting from an unproven approach. Relevant prior work includes INCLUDE, a large-scale isolated-sign dataset for ISL built by IIT Madras (Sridhar et al., 2020); CISLR, a broader-vocabulary word-level corpus (Joshi et al., 2022); ISL-CSLTR, a sentence-level continuous ISL dataset (Elakkiya and Natarajan, 2021); and OpenHands, a library of pose-based pretrained models benchmarked across ISL and other sign languages (Selvaraj et al., 2022). On the language side, recent research on gloss-to-text translation demonstrates that a language model can reconstruct fluent sentences from raw sign sequences, including dedicated correction stages that fix word order, omissions, and substitutions in draft translations. This project applies the same principle at a smaller, demo-appropriate scale.

## 6. System Architecture

The system is organised as a four-stage pipeline.

- Video capture: the signer performs a sequence of individual signs in front of a laptop webcam.

- Keypoint extraction: hand, body, and face landmarks are extracted from each frame, reducing each frame to a compact numeric representation rather than raw pixels.

- Sign recognition: a sequence model classifies each segment of keypoints into a word from the defined vocabulary, along with a confidence score.

- Sentence reconstruction: the resulting word sequence, tagged with confidence scores, is passed to a language model that reconstructs it into a natural, grammatically correct sentence, treating low-confidence words as open to reinterpretation and high-confidence words as fixed content.

The interface displays both the raw recognised word sequence and the final reconstructed sentence side by side, so the contribution of the language-model stage is visible to an audience.

## 7. Dataset Strategy

Recording a full training dataset from scratch was ruled out as impractical for the timeline. Instead, the project uses a combination of an existing public dataset and a small amount of targeted recording.

| Dataset | Vocabulary | Samples | Role in this project |
| --- | --- | --- | --- |
| INCLUDE (IIT Madras) | 263 words, 15 categories | 4,287 videos (~16 per word) | Primary training source |
| CISLR | 4,765 words | 7,050 videos (~1.5 per word) | Vocabulary reference only |
| ISL-CSLTR | 1,036 words / 100 sentences | 700 videos | Reference for sentence structure |
| Team-recorded data | Words absent from INCLUDE | Recorded by all five members | Fills vocabulary gaps; validates live performance |

The final vocabulary list is drawn first from INCLUDE, keeping the project grounded in an established, peer- reviewed resource. Words needed for the hospital and reception scenario that are absent from INCLUDE are recorded separately by the team. A small validation set recorded by team members and outside volunteers is used to confirm that the model performs reliably on signers who were not part of the main training data, which is essential for a credible live demonstration.

## 8. Model Approach

## 8.1 Sign Recognition


The recognition stage uses a sequence model over extracted keypoints, which is standard practice for isolated sign recognition at this vocabulary size and trains substantially faster than a model working directly on raw video. Where possible, the team fine-tunes an existing pretrained pose-based model rather than training from scratch, since published baselines on INCLUDE already achieve strong accuracy and fine-tuning reduces training time considerably. A simpler feature-based classifier is prepared in parallel as a fallback, so a working system exists even if the primary model needs further tuning.

## 8.2 Sentence Reconstruction

The language-model stage receives the ordered, confidence-tagged word sequence and is guided by a small set of worked examples showing typical inputs and the corresponding corrected sentences for this vocabulary and scenario. It is explicitly instructed to preserve all high-confidence content and not introduce new factual details, so that its role remains restricted to grammar and structure rather than invented meaning.

## 9. Team Structure

| Member | Role | Responsibility |
| --- | --- | --- |
| [Member 1] | Data Lead | Sources and organises training data; coordinates recording of words not covered by the public dataset |
| [Member 2] | ML Engineer – Recognition | Builds the keypoint extraction and sign-classification pipeline |
| [Member 3] | ML Engineer – Language Layer | Designs the prompt and correction logic that turns recognised words into a finished sentence |
| [Member 4] | Frontend & Integration | Builds the webcam capture interface and connects it to the recognition and language layers |
| [Member 5] | Integration Lead & Presentation | Owns end-to-end testing, bug triage, the demo script, and the final presentation |

## 10. Work Plan

| Timeline | Milestone |
| --- | --- |
| Day 1 | Vocabulary and scenario finalised; roles assigned; existing dataset shortlisted |
| Day 2 – 4 | Dataset prepared: public dataset downloaded and filtered to the chosen vocabulary; remaining words recorded by the team; keypoint extraction pipeline and webcam interface skeleton built in parallel |
| Day 5 | Preprocessing completed: keypoints cleaned, labelled, and split by signer |
| Day 6 – 7 | Recognition model trained and validated; language-model prompt designed and tested against realistic, imperfect word sequences |
| Day 8 – 9 | First full pipeline integration: webcam feed through to displayed sentence |
| Day 10 – 11 | Testing with signers outside the original recording group; threshold tuning; unreliable words identified and addressed |
| Day 12 | Feature freeze; interface polish; demo word set finalised |
| Day 13 | Backup demo recording completed; presentation deck built |
| Day 14 | Final rehearsal of live demo and backup, and presentation delivery |


## 11. Evaluation Plan

- Accuracy measured on signers held out from training, not only on the training group.

- A confusion analysis to identify visually similar signs that are frequently mistaken for one another.

- Sentence-level quality of the language-model output, judged against whether the reconstructed sentence preserves the intended meaning.

- End-to-end response time from completed signing to displayed sentence, kept short enough for a live demonstration.

## 12. Demonstration Plan

The presentation opens with a short live demonstration in which a team member signs a short sequence relevant to the hospital and reception scenario. A pre-recorded backup demonstration is available and used immediately if the live version does not perform as expected, rather than debugging on stage. The core of the presentation is a direct comparison between the raw recognised word sequence and the finished sentence produced by the language-model stage, since this contrast is the clearest illustration of the project's contribution.

## 13. Limitations

- The system recognises a fixed vocabulary of 100–150 words rather than open-vocabulary ISL.

- Signs are recognised individually rather than as continuous, fluent signing.

- Accuracy is expected to vary across signers and will be reported honestly rather than only under favourable conditions.

## 14. Future Scope

- Expand the vocabulary using additional public datasets and further recording.

- Move from isolated-sign recognition to continuous sentence-level recognition.

- Extend the system to mobile and edge devices for real-world deployment.

- Add regional dialect handling and a text-to-ISL reverse mode for two-way communication.

## 15. Conclusion

This project delivers a focused, working demonstration of ISL-to-text translation built on an established public dataset and a language-model correction stage that turns raw sign recognition into readable, natural sentences. The scope has been deliberately limited to what a five-person team can build and validate with confidence in two weeks, while the underlying architecture is designed to extend naturally toward a larger vocabulary and continuous signing in future work.

## 16. References

Sridhar, A., Ganesan, R. G., Kumar, P., & Khapra, M. M. (2020). INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition. Proceedings of the 28th ACM International Conference on Multimedia.

Joshi, A., Bhat, A., S, P., Gole, P., Gupta, S., Agarwal, S., & Modi, A. (2022). CISLR: Corpus for Indian Sign Language Recognition. Proceedings of EMNLP 2022.

Elakkiya, R., & Natarajan, B. (2021). ISL-CSLTR: Indian Sign Language Dataset for Continuous Sign Language Translation and Recognition. Mendeley Data.


Selvaraj, P., Nc, G., Kumar, P., & Khapra, M. (2022). OpenHands: Making Sign Language Recognition Accessible with Pose-based Pretrained Models across Languages. Proceedings of ACL 2022.

Joshi, A., et al. (2023). ISLTranslate: Dataset for Translating Indian Sign Language.

Fayyazsanavi, P., et al. (2024). Gloss2Text: Sign Language Gloss Translation using LLMs and Semantically Aware Label Smoothing.
