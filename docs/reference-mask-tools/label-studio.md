# Generating a Reference Mask with Label Studio

Label Studio provides a web-based annotation interface with optional SAM2 ML backend integration for interactive segmentation.

## Installation

```bash
pip install label-studio
label-studio start
```

Label Studio runs at `http://localhost:8080`.

## Project setup

Create a new project and use the following labeling interface config under **Settings → Labeling Interface → Custom template**:

```xml
<View>
  <Image name="image" value="$image"/>
  <BrushLabels name="tag" toName="image">
    <Label value="right_forewing" background="#FF0000"/>
    <Label value="left_forewing" background="#00FF00"/>
    <Label value="right_hindwing" background="#0000FF"/>
    <Label value="left_hindwing" background="#FFFF00"/>
  </BrushLabels>
  <KeyPointLabels name="tag2" toName="image" smart="true">
    <Label value="Foreground" smart="true" background="#FFaa00" showInline="true"/>
    <Label value="Background" smart="true" background="#00aaFF" showInline="true"/>
  </KeyPointLabels>
  <RectangleLabels name="tag3" toName="image" smart="true">
    <Label value="Foreground" smart="true" background="#FFaa00" showInline="true"/>
  </RectangleLabels>
</View>
```

## Manual brush annotation

Brush annotation works without any ML backend:

1. Open a task in your project
2. Select a wing label (e.g. `right_forewing`) from the label panel
3. Use the **Brush** tool to paint over that wing
4. Repeat for each of the four wings
5. Click **Submit** to save the annotation

## SAM2 ML backend (optional)

A SAM2 backend can be connected for interactive prompting:

1. Clone the backend repo: `git clone https://github.com/HumanSignal/label-studio-ml-backend`
2. Follow the SAM2 backend setup instructions in that repo
3. Start the backend on port 9090
4. In Label Studio go to **Settings → Machine Learning → Add Model** and enter `http://localhost:9090`

> **Known issue:** Label Studio Community Edition sends `"context": null` to ML backends instead of passing click coordinates. This means interactive prompting (keypoint/rectangle → SAM2 mask) does not currently work. Brush annotation works normally. Resolution requires either a Docker-based backend setup or a GPU backend on Cardinal.

## Current status

Setup complete. Manual brush annotation is functional. Interactive SAM2 prompting is pending resolution of the `context: null` issue.
