# LLM Notes

## Mistral ECFR


### Full Fine Tuning Statistics
**Training 1**
```py
{
	'eval_loss': 2.9787986278533936,
	'eval_runtime': 0.1087,
	'eval_samples_per_second': 27.604,
	'eval_steps_per_second': 9.201,
	'epoch': 0.9090909090909091
}
```

**Training 2**
```py
{
	'eval_loss': 1.6337567567825317,
 	'eval_runtime': 0.1547,
 	'eval_samples_per_second': 19.394,
 	'eval_steps_per_second': 6.465,
 	'epoch': 2.5454545454545454
 }
```

### Chat 1 - Full Fine Tune
>Obviously catastrophic overfitting with massive forgetting

```
Prompt: Hello


##### 2.1

##### § 2.1 Official emblem. #####

The official emblem of the Department is a gold-framed star with the letters “DEPARTMENT” written in the center in letters that are smaller than the star. The star has the letters “UNITED STATES OF AMERICA” at its base, in letters that are smaller than the star. The star is enclosed in a gold border with the letters “UNITED STATES OF AMERICA” in letters that are smaller than the star and in smaller letters than the star. The letters in the official emblem are in letters of the same size as the letters in the official seal of the United States. The letters in the official emblem are in letters of the same size as the letters in the official seal of the United States. The letters in the official emblem are in letters of the same size as the letters in the official seal of the United States. The letters in the official emblem are in letters of the same size as the letters in the official seal of the United States. The letters in the official emblem are in letters of the same size as the letters in the official seal of the United States. The letters in the official emblem are in letters of the same size as the letters in the official seal of the United States. The letters in the official emblem are in letters of the same size as the letters in the official seal of the United States. The letters in the official emblem are in letters of the same size as the letters in the official seal of the
> Crypto
 26

##### 2.2

##### § 2.2 Official seal. #####

The official seal of the Department is in the shape of a star with letters in the
```

I began testing to see if it could recite simple information from the ecfr, namely, the first section "Definitions".

## Chat 2 - Light Lora Training 60 steps
> Incorrect information, wrong structure, wrong titles, and forgetting of section 1.1 Definitions
```
Prompt: Recite Title 1 Section 1.1 of the eCFR
 Title 1 of the eCFR (Electronic Code of Federal Regulations) does not contain Section 1.1. Title 1 of the eCFR is related to General Provisions, and it starts with Section 1.0. However, the specific content you're asking for might be found in a different title or section of the eCFR. Here's the content of Title 1, Section 1.0:

Title 1—General Provisions

Subchapter A—Office of the Federal Register (Continued)

PART 1—GENERAL

Subpart A—General Provisions

Authority: 5 U.S.C. 301.

Sec. 1.0  Scope.

This part is issued by the Office...
```

## Chat 3 - Light Lora Training 100 epochs
> Closer but still incorrect information
```
Prompt: Recite Title 1 Section 1.1 of the eCFR
 Title 1 of the Electronic Code of Federal Regulations (eCFR) is titled "General Provisions" and contains the following content for Section 1.1:

Title 1 - General Provisions

Subchapter A - General

Part 1 - General Provisions

Section 1.1 - Scope and purpose.

(a) The regulations in this part implement the Office of Management and Budget's (OMB) authority to establish government-wide standards for the organization, format, and content of rules and other documents issued by federal agencies.

(b) The regulations in this part are applicable to rules and other documents issued by federal agencies, regardless of the legal source or authority under which the documents are issued, except as provided in paragraph...
```

## Chat 4 - Light Lora Training 200 epochs
> Unstable, forgetfulness, odd character outputs
```
> Recite Title 1 Section 1.1 of the eCFR
 Title 1, Section 1.1 of the eCFR (Code of Federal Regulations) reads:

"Part 1

[]

"Title

"Electronic Document Procedures for Federal Agencies

"Part 1

" eCFR

"

"This document represents a copy of Title 1 of the Code of Federal Regulations (CFR) as it appears on a public electronic dissemination service which is made available free of charge by the Government Publishing Office. This service is updated and published each day that a final rule is published in Title 1 of the CFR. The text of this service includes all material that appears in the print version of the CFR, including editorial notes and appendices, and it is amended each month that an editorial note
```