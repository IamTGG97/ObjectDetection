import cv2

def is_inside(inner, outer, threshold=0.5):
    #checks if inner bounding box is inside outer bounding box 
    #threshold means that 0.5 = at least 50% of the inner box must overlap with the outer bounding box

    ix1, iy1, ix2, iy2 = inner
    #unpacks inner box coordinates

    ox1, oy1, ox2, oy2 = outer
    #unpacks outer box coordinates

    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)
    #calculates intersection coordinates

    if inter_x2 < inter_x1 or inter_x2 < inter_y1:
        return False
    #if no overlap return false

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    inner_area = (ix2 - ix1) * (iy2 - iy1)
    #calculates area of intersection and inner box

    return (inter_area/inner_area) >= threshold
    #returns true if enough of te inner box is inside the outer box

def build_scene_description(results, model):
    #takes yolo results and the model and builds a description of the scene based on the detected objects and their relationships

    boxes = results[0].boxes #gets bounding boxes from results

    labels, coords = [], []

    for box in boxes:
        label = model.names[int(box.cls)]
        #gets class name for the detection

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        #gets the bounding box coordinates

        labels.append(label)
        coords.append((x1, y1, x2, y2))
        #stores labels and coordinates in separate lists

    if not labels:
        return "Nothing detected"
    #if no objects detected return this

    person_boxes = [(c, i) for i, (l, c) in enumerate(zip(labels, coords)) if l == "person"]
    #finds all detected person boxes and their index positions

    non_person = [(labels[i], coords[i]) for i in range(len(labels)) if labels[i] != "person"]
    #gets all objects that are not people

    if not person_boxes:
        #no people in frame, just list non-person objects

        unique = list(set([l for l, _ in non_person]))
        #gets unique non-person object types to avoid repetition

        if len(unique) == 1:
            return f"A {unique[0]} is visible"
        elif len(unique) == 2:
            return f"A {unique[0]} and a {unique[1]} are visible"
        else:
            items = ", ".join([f"a {u}" for u in unique[:-1]])
            return f"{items}, and a {unique[-1]} are visible"
        #builds sentence for one or multiple items

    held_items = []
    for person_coord, _ in person_boxes:
        for obj_label, obj_coord in non_person:
            if is_inside(obj_coord, person_coord):
                held_items.append(obj_label)

    #checks each object against each person box to see if it is being held by a person using the is_inside function

    loose_items = [l for l, c in non_person if not any(
        is_inside(c, pc) for pc, _ in person_boxes
    )]
    #finds items that are visible but not inside persons bounding boxes
    
    person_count = len(person_boxes)
    person_str = "A person" if person_count == 1 else f"{person_count} people"
    #builds person count string

    description = person_str

    if held_items:
        unique_held = list(set(held_items))
        #removes duplicate held items labels

        if len(unique_held) == 1:
            description += f" holding a {unique_held[0]}"
        elif len(unique_held) == 2:
            description += f" holding a {unique_held[0]} and a {unique_held[1]}"
        else:
            items = ", ".join([f"a {h}" for h in unique_held[:-1]])
            description += f" holding {items}, and a {unique_held[-1]}"
        #corrects grammar for multiple items held

    if loose_items:
        unique_loose = list(set(loose_items))
        loose_str = ", ".join([f"a {l}" for l in unique_loose])
        description += f" near {loose_str}"
        #appends nearby objects that arent inside other bounding boxes

    return description

def draw_description_bar(frame, description):
    #draws transparent black bar at the bottom of the frame with white text

    overlay = frame.copy()

    h, w = frame.shape[:2] #gets height and width

    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1) #draws black rectangle at bottom of frame

    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame) #adds transparency to rectangle

    cv2.putText(frame, description, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2) #white text

    return frame
