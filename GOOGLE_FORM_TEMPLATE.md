# Google Form Question Template

## Form Title
**CCSD v2.0 Dataset Access Request**

## Form Description
Thank you for your interest in the Chinese Compositional Safety Dataset (CCSD) v2.0.

Due to the sensitive nature of content moderation data involving geopolitical topics, this dataset is available for academic research purposes only. Please complete this form to request access.

Approval typically takes 3-5 business days. You will receive download instructions via email upon approval.

**Before proceeding, please read the [Data Use Agreement](link-to-agreement).**

---

## Questions

### Section 1: Personal Information

**Q1. Full Name** *(Required)*
- Type: Short text
- Validation: Non-empty

**Q2. Institutional Email** *(Required)*
- Type: Short text
- Validation: Email format, must not be personal email (Gmail, QQ, 163, etc.)
- Help text: "Must be your university or research institution email address"

**Q3. Confirm Email** *(Required)*
- Type: Short text
- Validation: Must match Q2

**Q4. Current Position** *(Required)*
- Type: Multiple choice
- Options:
  - [ ] PhD Student
  - [ ] Master's Student
  - [ ] Postdoctoral Researcher
  - [ ] Faculty / Professor
  - [ ] Research Scientist
  - [ ] Other (please specify)

**Q5. Institution / University** *(Required)*
- Type: Short text
- Help text: "Full name of your university or research institution"

**Q6. Department / Lab** *(Required)*
- Type: Short text

**Q7. Country** *(Required)*
- Type: Dropdown (list of countries)

---

### Section 2: Supervisor Information (for students)

**Q8. Supervisor / PI Name** *(Required for students)*
- Type: Short text
- Condition: Show if Q4 = PhD Student OR Master's Student

**Q9. Supervisor Email** *(Required for students)*
- Type: Short text
- Validation: Email format
- Condition: Show if Q4 = PhD Student OR Master's Student

---

### Section 3: Research Plan

**Q10. Research Purpose** *(Required)*
- Type: Paragraph text
- Character limit: 500
- Help text: "Please describe how you plan to use the dataset (min. 100 characters)"

**Q11. Intended Publication Venue** *(Optional)*
- Type: Short text
- Help text: "e.g., ACL 2026, EMNLP 2026, etc."

**Q12. Have you read the Data Use Agreement?** *(Required)*
- Type: Multiple choice
- Options:
  - [ ] Yes, I have read and agree to the terms
  - [ ] No

**Q13. Do you agree to cite the paper if you use the dataset?** *(Required)*
- Type: Multiple choice
- Options:
  - [ ] Yes, I agree to cite the paper
  - [ ] No

**Q14. How did you hear about this dataset?** *(Optional)*
- Type: Multiple choice
- Options:
  - [ ] Academic paper
  - [ ] Conference presentation
  - [ ] Social media
  - [ ] Colleague recommendation
  - [ ] Other (please specify)

---

### Section 4: Confirmation

**Q15. Certification** *(Required)*
- Type: Checkboxes
- Options (all must be checked):
  - [ ] I confirm that the information provided is accurate
  - [ ] I will use the dataset solely for academic research purposes
  - [ ] I will not redistribute the dataset
  - [ ] I will comply with all applicable laws and institutional policies
  - [ ] I understand that violation of terms may result in access revocation

---

## Confirmation Message

```
Thank you for your request!

We have received your application for CCSD v2.0 dataset access.

Our team will review your request within 3-5 business days. If approved,
you will receive an email with download instructions.

If you have questions, please contact: 230239771@seu.edu.cn

Research Team
Southeast University
```

---

## Email Notification Template

**Subject:** [CCSD v2.0] Request Status Update

**Body (Approved):**
```
Dear [Name],

Your request for CCSD v2.0 dataset access has been APPROVED.

Download Link: [secure link, valid for 7 days]
Documentation: [link]
Sample Code: [link]

Please note:
- Do not share this link with others
- Each researcher must submit their own request
- Remember to cite our paper if you use this dataset

If you have questions, please contact us.

Best regards,
Research Team
Southeast University
```

**Body (Rejected):**
```
Dear [Name],

Thank you for your interest in CCSD v2.0.

After review, we are unable to approve your request at this time.
Common reasons include:
- Non-institutional email address
- Insufficient research description
- Ineligible requester type

If you believe this is an error, please contact us with additional
information.

Best regards,
Research Team
Southeast University
```
