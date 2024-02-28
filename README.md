### What is crowding?
Crowding is a common malocclusion that affects the oral health and aesthetics of many people. It happens when there is not enough space in the dental arch for the teeth to align
correctly.

<p align = "center"><img src='https://github.com/richiephang/Crowding/assets/76896343/21b5ac49-3812-4162-8305-fd9d5ce4828b' width='300'></p>
<br>

### Why we need to assess crowding severity?
Assessing crowding severity is one of the essential steps in orthodontic treatment planning, as it helps to determine the space required to align the teeth correctly before further treatment planning like tooth extraction, bracket placement etc.

### Problem statement
Conventional methods of assessing crowding severity are subjective and time-consuming, relying on visual examination, manual measurements or subjective judgements by orthodontists. There are no firm criteria or standardized method for assessing crowding severity.
<br><br>
### About this project
This project aims to propose a method that can automate the process of assessing crowding severity using intraoral photographs, improve the efficiency of orthodontic treatment planning.

#### Proposed method:
1. Tooth Landmarks Detection
   * Pretrained VGG-19 is used to apply transfer learning to train a tooth landmarks detection model.
   * The model can predict mesial and distal points for each tooth on intraoral photographs.
     
     <img src='https://github.com/richiephang/Crowding/assets/76896343/5ea4d326-a9fb-407b-a408-fe3a69add1a9' width='350'>

      
2. Arch Form Determination
   * B-spline curve fitting is applied to the predicted tooth points to form a smooth dental arch curve.
     
     <img src='https://github.com/richiephang/Crowding/assets/76896343/640309fc-a4eb-4f92-8d3d-6e87bf9a9bf4' width='350'>

3. Crowding Severity Assessment
   * To measure the tooth widths (mesiodistal width), the distance between the predicted mesial and distal points is calculated for each tooth in the intraoral photograph.
     
     <img src='https://github.com/richiephang/Crowding/assets/76896343/2d9c3d82-e385-4259-8fcf-98426b3daf90' width='350'>
     
   * However, the tooth widths calculated are in pixel unit. To convert the tooth widths to millimeters, scaling factor is calculated, which is the ratio of the pixel width to the millimeter width of the central incisor tooth. Millimeter width of the central incisor tooth should be measured and input by the user.
   * The severity of crowding is assessed by calculating the difference between sum of all tooth widths and arch form length. The higher the value, the more severe the crowding.




     
     
     
     

