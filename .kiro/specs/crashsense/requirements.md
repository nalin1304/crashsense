# Requirements Document

## Introduction

CrashSense is an end-to-end Acoustic Triangulation and Autonomous Drone Dispatch system that detects highway vehicle crashes from audio, localizes the crash position using Time-Difference-of-Arrival (TDOA) across three simulated roadside sensors, dispatches a virtual drone from the nearest toll plaza, and visualizes the entire flow on a real-time React dashboard backed by a FastAPI WebSocket server. The system spans five integrated phases: audio AI detector, TDOA triangulation, backend API, dashboard, and demo orchestration. The feature targets a live demonstration in which a single trigger drives a sub-three-second pipeline from raw audio to drone arrival animation.

## Glossary

- **CrashSense**: The complete system encompassing all components defined below.
- **Audio_Detector**: The PyTorch model and inference module that classifies audio segments as CRASH or NORMAL.
- **Spectrogram_Generator**: The component that converts raw audio waveforms into 224x224 mel spectrogram images.
- **Trainer**: The training pipeline that fits the Audio_Detector on labeled spectrograms.
- **Triangulation_Solver**: The TDOA module that performs forward simulation and inverse localization of crash coordinates.
- **Sensor_Config**: The static configuration of three simulated highway sensors with fixed geographic coordinates.
- **Backend_API**: The FastAPI application exposing REST endpoints and a WebSocket channel.
- **WebSocket_Manager**: The connection manager that maintains active WebSocket clients and broadcasts events.
- **Dashboard**: The React single-page application that visualizes sensors, crash events, and drone dispatch.
- **Map_View**: The Mapbox GL JS map component embedded in the Dashboard.
- **Drone_Tracker**: The Dashboard component that animates a drone icon from a toll plaza dock to the crash location.
- **Alert_Panel**: The right-side scrollable panel in the Dashboard that displays crash event cards.
- **Audio_Monitor**: The Web Audio API waveform visualizer component in the Dashboard.
- **WebSocket_Client**: The `useWebSocket` React hook that manages the connection from the Dashboard to the Backend_API.
- **Demo_Runner**: The orchestration script (`demo_runner.py`) that drives the end-to-end pipeline.
- **CrashEvent**: The Pydantic-defined event object broadcast on the WebSocket channel and rendered as alert cards.
- **TDOA**: Time Difference of Arrival, the technique used to localize the crash from per-sensor arrival timestamps.
- **Sensor_Triangle**: The triangular region bounded by the three sensors S1, S2, and S3.
- **Toll_Plaza**: A sensor location designated as a drone dock origin (S1 and S3 in this system).
- **SPEED_OF_SOUND**: The fixed acoustic propagation constant of 343.0 meters per second used by the Triangulation_Solver.

## Requirements

### Requirement 1: Audio Dataset Acquisition

**User Story:** As a machine learning engineer, I want a labeled audio dataset of crash and non-crash sounds, so that I can train the Audio_Detector to distinguish crashes from background highway noise.

#### Acceptance Criteria

1. THE CrashSense SHALL store raw audio clips as WAV files under `data/raw_audio/crash/` and `data/raw_audio/noise/` directories, with each stored clip having a duration of at least 3.0 seconds and at most 10.0 seconds.
2. THE CrashSense SHALL provide a dataset preparation script `backend/audio_model/dataset_prep.py` that downloads clips from the Google AudioSet ontology categories `Vehicle crash`, `Skidding`, and `Vehicle horn, car horn, honking` using the `audioset-download` package into the `crash` class.
3. THE CrashSense SHALL include Google AudioSet clips labeled `Traffic noise, roadway noise` and `Wind` in the `noise` class.
4. THE CrashSense SHALL acquire between 500 and 2000 clips for the `crash` class and between 500 and 2000 clips for the `noise` class, with the per-class clip count differing by no more than 10 percent between the two classes.
5. IF the `dataset_prep.py` AudioSet download terminates with an unhandled exception, exceeds 30 minutes of total runtime, or completes with fewer than 500 clips in either class, THEN THE CrashSense SHALL supplement that class from the ESC-50 dataset, mapping ESC-50 categories `car_horn`, `engine`, and `breaking_glass` to the `crash` class and ESC-50 categories `wind` and `rain` to the `noise` class, until each class contains at least 500 clips.

### Requirement 2: Spectrogram Generation

**User Story:** As a machine learning engineer, I want raw audio converted into fixed-size mel spectrogram images, so that I can train an image classifier on a uniform input representation.

#### Acceptance Criteria

1. THE Spectrogram_Generator SHALL be implemented in `backend/audio_model/spectrogram_gen.py`.
2. WHEN a wav file is provided to the Spectrogram_Generator and the file decodes successfully, THE Spectrogram_Generator SHALL produce a single mel spectrogram PNG image with width 224 pixels and height 224 pixels.
3. THE Spectrogram_Generator SHALL resample each input signal to 22050 Hz, downmix multi-channel audio to a single channel, compute a mel spectrogram with 128 mel bands and an upper frequency bound of 8000 Hz, and target a fixed clip duration of 3.0 seconds.
4. WHEN an input clip duration is shorter than 3.0 seconds, THE Spectrogram_Generator SHALL pad the clip with zero-amplitude samples to exactly 3.0 seconds before generating the spectrogram.
5. WHEN an input clip duration is longer than 3.0 seconds, THE Spectrogram_Generator SHALL trim the clip to the first 3.0 seconds before generating the spectrogram.
6. WHEN an input wav file located at `data/raw_audio/<class>/<name>.wav` is processed successfully, THE Spectrogram_Generator SHALL write the generated PNG to `data/spectrograms/<class>/<name>.png`, where `<class>` is `crash` or `noise`.
7. WHEN a target output directory under `data/spectrograms/` does not exist at write time, THE Spectrogram_Generator SHALL create the directory before writing the PNG.
8. IF a provided input file cannot be decoded as a valid wav signal, THEN THE Spectrogram_Generator SHALL skip that file, emit an error message identifying the file path and the failure reason, and continue processing any remaining files without writing a PNG for the skipped file.

### Requirement 3: Audio Classification Model Training

**User Story:** As a machine learning engineer, I want a trained binary classifier for crash detection, so that the Audio_Detector can be invoked at inference time with high accuracy.

#### Acceptance Criteria

1. THE Audio_Detector SHALL be defined in `backend/audio_model/model.py` as a torchvision ResNet-18 with a final fully-connected layer replaced by a 2-class head.
2. THE Audio_Detector SHALL be initialized with pretrained ImageNet weights from torchvision.
3. THE Trainer SHALL be implemented in `backend/audio_model/train.py` and SHALL load spectrograms via `torchvision.datasets.ImageFolder` rooted at `data/spectrograms/`.
4. THE Trainer SHALL split the dataset into 80 percent training and 20 percent validation partitions using `torch.utils.data.random_split` with a deterministic generator seed of 42.
5. THE Trainer SHALL apply RandomHorizontalFlip with probability 0.5 and ColorJitter with brightness 0.2, contrast 0.2, saturation 0.2, and hue 0.05 to training samples only.
6. THE Trainer SHALL not apply vertical flip augmentation to any sample.
7. THE Trainer SHALL use the Adam optimizer with learning rate 1e-4 and weight decay 1e-4.
8. THE Trainer SHALL use CrossEntropyLoss for optimization.
9. THE Trainer SHALL run for 20 epochs with a training batch size of 32 and a validation batch size of 32.
10. WHEN each epoch completes, THE Trainer SHALL print the validation accuracy for that epoch as a percentage rounded to two decimal places.
11. THE Trainer SHALL save the checkpoint with the highest validation accuracy across all 20 epochs to `backend/audio_model/crash_detector.pth`.
12. THE Trainer SHALL achieve at least 85 percent validation accuracy on the saved best checkpoint produced by criterion 11.
13. IF the saved best checkpoint achieves less than 80 percent validation accuracy after the initial training run, THEN THE Trainer SHALL replace the backbone with a `timm` EfficientNet-B0 backbone, repeat training exactly once using the same dataset split, augmentations, optimizer, loss, batch size, and epoch count, and overwrite `backend/audio_model/crash_detector.pth` with the new best checkpoint from the fallback run.

### Requirement 4: Audio Inference

**User Story:** As a system integrator, I want a fast in-memory inference function for crash detection, so that audio streams can be classified in real time without disk I/O.

#### Acceptance Criteria

1. THE Audio_Detector SHALL expose an inference entry point named `predict` in `backend/audio_model/inference.py` that accepts a single positional argument and returns a dictionary as specified in criterion 6.
2. THE Audio_Detector SHALL accept the input argument as either a string filesystem path to a WAV file or a one-dimensional NumPy float32 array of mono PCM samples at the 22050 Hz sample rate defined in Requirement 2.
3. WHEN audio is provided to the Audio_Detector at inference time, THE Audio_Detector SHALL convert the audio to the 224 by 224 mel spectrogram representation defined in Requirement 2 entirely in memory and SHALL NOT write any intermediate file to disk.
4. WHEN the Audio_Detector is initialized, THE Audio_Detector SHALL load model weights from `backend/audio_model/crash_detector.pth` exactly once and reuse the loaded model for all subsequent `predict` calls within the same process.
5. IF the weights file `backend/audio_model/crash_detector.pth` is missing or fails to deserialize, THEN THE Audio_Detector SHALL raise a runtime error indicating that the crash detector checkpoint could not be loaded and SHALL NOT return a prediction.
6. WHEN `predict` completes successfully, THE Audio_Detector SHALL return a JSON-serializable dictionary containing the field `event` equal to `"CRASH"` when the softmax probability of the crash class is greater than or equal to 0.5 and equal to `"NORMAL"` otherwise, and the field `confidence` equal to the softmax probability of the predicted class as a float in the inclusive range 0.0 to 1.0.
7. WHILE processing a continuous audio stream longer than 3.0 seconds, THE Audio_Detector SHALL apply a sliding window of length 3.0 seconds with a hop of 0.5 seconds and SHALL emit one prediction dictionary per window in chronological order.
8. WHEN the demo dashcam crash audio is processed and a window prediction yields `event` equal to `"CRASH"` with `confidence` greater than or equal to 0.85, THE Audio_Detector SHALL print the line `CRASH DETECTED — confidence: <value>%` exactly once for that window, where `<value>` is the `confidence` value multiplied by 100 and rounded to one decimal place using banker-free half-up rounding.

### Requirement 5: Sensor Configuration

**User Story:** As a developer, I want a fixed three-sensor configuration with named highway locations, so that the Triangulation_Solver and the Map_View share a single source of truth for sensor coordinates.

#### Acceptance Criteria

1. THE Sensor_Config SHALL be defined in `backend/triangulation/sensor_config.py` and SHALL expose exactly three sensor records and the `SPEED_OF_SOUND` constant as importable Python module-level attributes.
2. THE Sensor_Config SHALL define sensor S1 with stable identifier `"S1"`, display name `"Toll Plaza Alpha"`, latitude 19.1136 decimal degrees, and longitude 72.8697 decimal degrees in the WGS84 coordinate reference system.
3. THE Sensor_Config SHALL define sensor S2 with stable identifier `"S2"`, display name `"CCTV Pole B12"`, latitude 19.1089 decimal degrees, and longitude 72.8812 decimal degrees in the WGS84 coordinate reference system.
4. THE Sensor_Config SHALL define sensor S3 with stable identifier `"S3"`, display name `"Toll Plaza Beta"`, latitude 19.1201 decimal degrees, and longitude 72.8754 decimal degrees in the WGS84 coordinate reference system.
5. THE Sensor_Config SHALL designate sensors S1 and S3 as Toll_Plaza locations and sensor S2 as a non-Toll_Plaza location, such that an enumeration of drone dispatch origins returns exactly the two sensor records identified by `"S1"` and `"S3"`.
6. THE Sensor_Config SHALL expose a constant `SPEED_OF_SOUND` set to 343.0 meters per second.

### Requirement 6: Forward TDOA Simulation

**User Story:** As a developer, I want to simulate per-sensor acoustic arrival times for a known crash coordinate, so that I can validate the inverse Triangulation_Solver against ground truth.

#### Acceptance Criteria

1. THE Triangulation_Solver SHALL provide a function `simulate_arrival_times(crash_lat, crash_lon, sensors)` in `backend/triangulation/tdoa_solver.py` that accepts `crash_lat` and `crash_lon` as floating-point latitude and longitude values in decimal degrees and `sensors` as an iterable of sensor records, where each record exposes a string `name`, a floating-point latitude in decimal degrees, and a floating-point longitude in decimal degrees consistent with the Sensor_Config.
2. WHEN `simulate_arrival_times` is called with a crash coordinate and the Sensor_Config, THE Triangulation_Solver SHALL compute the geodesic distance in meters from `(crash_lat, crash_lon)` to each sensor's coordinate using `geopy.distance.geodesic`.
3. WHEN `simulate_arrival_times` completes successfully, THE Triangulation_Solver SHALL return a dictionary containing exactly one entry per input sensor, with each key set to that sensor's `name` string and each value set to a non-negative floating-point arrival time in seconds equal to that sensor's geodesic distance in meters divided by `SPEED_OF_SOUND`.

### Requirement 7: Inverse TDOA Localization

**User Story:** As a system integrator, I want to recover crash coordinates from per-sensor arrival times, so that the Backend_API can localize real or simulated crash events from sensor data alone.

#### Acceptance Criteria

1. THE Triangulation_Solver SHALL provide a function `tdoa_localize(time_delays, sensors)` in `backend/triangulation/tdoa_solver.py` that accepts `time_delays` as a mapping from sensor name to arrival time in seconds and `sensors` as the Sensor_Config sensor collection.
2. THE Triangulation_Solver SHALL designate the sensor whose entry in `time_delays` has the smallest arrival time as the reference sensor and SHALL solve the TDOA equations using `scipy.optimize.fsolve` with least-squares residuals computed relative to that reference sensor.
3. THE Triangulation_Solver SHALL initialize the `scipy.optimize.fsolve` search at the centroid of the Sensor_Triangle and SHALL accept only solutions whose latitude and longitude lie inside the axis-aligned bounding box of the Sensor_Triangle.
4. WHEN `tdoa_localize` completes with an accepted solution, THE Triangulation_Solver SHALL return a dictionary with key `lat` as a float in the inclusive range -90.0 to 90.0 and key `lon` as a float in the inclusive range -180.0 to 180.0.
5. IF `scipy.optimize.fsolve` fails to converge or yields a coordinate outside the bounding box of the Sensor_Triangle, THEN THE Triangulation_Solver SHALL signal localization failure to the caller in a manner distinguishable from a successful result and SHALL NOT return values for `lat` or `lon`.
6. WHEN forward-simulated arrival times perturbed by independent Gaussian noise with standard deviation 2 milliseconds are passed to `tdoa_localize` for a crash point selected uniformly at random inside the Sensor_Triangle, THE Triangulation_Solver SHALL return a coordinate within 30 meters of the true crash coordinate for at least 95 of 100 independent trials.
7. THE CrashSense SHALL provide a demonstration script `tdoa_demo.py` at the repository root that selects a uniformly random crash point inside the Sensor_Triangle, computes true arrival times via `simulate_arrival_times`, adds independent Gaussian noise with standard deviation 2 milliseconds to each per-sensor arrival time, invokes `tdoa_localize`, and prints the true coordinate, the solved coordinate, and the localization error in meters to at least one decimal place.

### Requirement 8: Geographic Utility Functions

**User Story:** As a developer, I want shared geographic helper functions, so that distance, bearing, and offset computations remain consistent across the Triangulation_Solver and the Drone_Tracker.

#### Acceptance Criteria

1. THE CrashSense SHALL provide `backend/triangulation/geo_utils.py` exporting the functions defined in criteria 2 through 4.
2. WHEN `meters_to_latlon_offset(north_meters, east_meters, reference_lat)` is called with northward and eastward offsets in meters and a reference latitude in decimal degrees, THE CrashSense SHALL return a tuple `(delta_lat, delta_lon)` in decimal degrees that, when added to a coordinate at the reference latitude, displaces that coordinate by the supplied offsets within 1 meter accuracy for absolute offset magnitudes up to 10000 meters.
3. WHEN `bearing_between_two_points(lat1, lon1, lat2, lon2)` is called with two coordinates in decimal degrees, THE CrashSense SHALL return the initial geodesic bearing from point 1 to point 2 as a float in the inclusive range 0.0 to 360.0 degrees measured clockwise from true north.
4. WHEN `nearest_sensor_to_point(lat, lon, sensors)` is called with a coordinate in decimal degrees and a non-empty list of Sensor_Config entries, THE CrashSense SHALL return the sensor whose `geopy.distance.geodesic` distance to `(lat, lon)` is the smallest, breaking ties by selecting the sensor with the lexicographically smallest name.
5. IF `nearest_sensor_to_point` is called with an empty sensor list, THEN THE CrashSense SHALL raise an error indicating that the sensor list must contain at least one sensor.

### Requirement 9: Backend API Application

**User Story:** As a frontend developer, I want a FastAPI backend with CORS enabled, so that the Dashboard running on a different port can call REST and WebSocket endpoints without browser restrictions.

#### Acceptance Criteria

1. THE Backend_API SHALL be implemented as a FastAPI application with its entry module located at `backend/api/main.py`.
2. THE Backend_API SHALL configure CORS to allow requests from any origin, permit all HTTP methods, and permit all request headers, so that a Dashboard served from any localhost port can issue cross-origin REST and WebSocket calls without browser-imposed CORS rejection or preflight failure.
3. THE Backend_API SHALL include the router defined in `backend/api/routes.py` such that every REST and WebSocket endpoint declared in that router is reachable through the FastAPI application.
4. WHEN launched with uvicorn, THE Backend_API SHALL begin accepting HTTP and WebSocket connections on port 8000 within 10 seconds of process start.
5. IF port 8000 is unavailable at startup, THEN THE Backend_API SHALL terminate with a non-zero exit code and emit a startup error message indicating that port 8000 cannot be bound.

### Requirement 10: WebSocket Connection Management

**User Story:** As a backend developer, I want a connection manager that maintains active WebSocket clients, so that crash events can be broadcast to all connected dashboards simultaneously.

#### Acceptance Criteria

1. THE WebSocket_Manager SHALL be implemented as a class in `backend/api/ws_manager.py`.
2. THE WebSocket_Manager SHALL maintain an in-memory collection of active WebSocket connections supporting up to 500 concurrent connections.
3. WHEN a client successfully completes the WebSocket handshake on the WebSocket endpoint, THE WebSocket_Manager SHALL add the connection to the active connection collection before any message is dispatched to that client.
4. WHEN a client disconnects from the WebSocket endpoint through either a normal close handshake or an abnormal disconnect such as a network error or send failure, THE WebSocket_Manager SHALL remove the connection from the active connection collection.
5. WHEN the `broadcast(message: dict)` method is invoked with a non-empty active connection collection, THE WebSocket_Manager SHALL serialize the message as JSON once and send the serialized payload to every connection currently in the active connection collection.
6. IF sending the broadcast payload to an individual connection fails or does not complete within 5 seconds, THEN THE WebSocket_Manager SHALL remove that connection from the active connection collection and continue sending to the remaining connections without raising an error to the caller of `broadcast(message: dict)`.
7. IF the `broadcast(message: dict)` method is invoked while the active connection collection is empty, THEN THE WebSocket_Manager SHALL return without performing any send operation and without raising an error.

### Requirement 11: REST and WebSocket Endpoints

**User Story:** As a frontend developer, I want documented REST and WebSocket endpoints, so that the Dashboard can fetch sensor metadata, trigger simulations, and subscribe to live crash events.

#### Acceptance Criteria

1. WHEN a client issues a GET request to `/sensors`, THE Backend_API SHALL return, within 1 second, the three sensor entries from the Sensor_Config, where each entry includes the sensor name (1 to 64 characters), latitude (-90.0 to 90.0), and longitude (-180.0 to 180.0).
2. WHEN `POST /simulate-crash` is called with a JSON body containing `lat` (-90.0 to 90.0) and `lon` (-180.0 to 180.0), THE Backend_API SHALL execute the TDOA forward simulation, then the inverse localization, then broadcast a CrashEvent to all `/ws/events` subscribers, and return an HTTP success response within 5 seconds.
3. IF `POST /simulate-crash` is called with a missing or non-numeric `lat` or `lon`, or with values outside the ranges -90.0 to 90.0 (lat) or -180.0 to 180.0 (lon), THEN THE Backend_API SHALL reject the request with a client-error response indicating the invalid field, SHALL NOT execute the pipeline, and SHALL NOT broadcast a CrashEvent.
4. WHEN `POST /detect-audio` is called with an uploaded audio file no larger than 10 MB in WAV or MP3 format, THE Backend_API SHALL invoke the Audio_Detector and return, within 10 seconds, a JSON response containing the inference label and a confidence score between 0.0 and 1.0.
5. IF `POST /detect-audio` is called with no file, an unsupported format, or a file larger than 10 MB, THEN THE Backend_API SHALL reject the request with a client-error response indicating the validation failure and SHALL NOT invoke the Audio_Detector.
6. WHEN a client connects to the WebSocket endpoint `/ws/events`, THE Backend_API SHALL accept the connection and SHALL push every CrashEvent generated thereafter to that client within 1 second of generation, until the client disconnects.
7. WHEN a client issues a GET request to `/drone-status` while a dispatch animation is active, THE Backend_API SHALL return, within 1 second, the current drone latitude (-90.0 to 90.0), longitude (-180.0 to 180.0), and a dispatch status field with value `idle`, `in_transit`, or `arrived`.
8. IF a client issues a GET request to `/drone-status` while no dispatch animation is active, THEN THE Backend_API SHALL return a response with the dispatch status field set to `idle` and the last known drone coordinates.

### Requirement 12: Crash Event Schema

**User Story:** As a frontend developer, I want a strongly typed CrashEvent schema, so that the Dashboard can render every crash card with consistent fields.

#### Acceptance Criteria

1. THE CrashSense SHALL define the CrashEvent Pydantic model in `backend/api/schemas.py` with all fields declared as required (non-optional) unless explicitly marked otherwise in this requirement.
2. THE CrashEvent SHALL include the fields `event_id` (string, UUID v4 format, exactly 36 characters), `timestamp` (ISO 8601 UTC datetime string), `crash_lat` (float, range -90.0 to 90.0 inclusive), `crash_lon` (float, range -180.0 to 180.0 inclusive), `confidence` (float, range 0.0 to 1.0 inclusive), `nearest_sensor` (string, 1 to 64 characters), `drone_origin_lat` (float, range -90.0 to 90.0 inclusive), `drone_origin_lon` (float, range -180.0 to 180.0 inclusive), `eta_seconds` (non-negative integer, range 0 to 86400 inclusive), and `status` (string enum).
3. THE CrashEvent `status` field SHALL accept exactly one of the values `"DETECTED"`, `"DRONE_DISPATCHED"`, or `"DRONE_ARRIVED"`.
4. IF a CrashEvent instance is constructed with any field missing, of the wrong type, or outside the specified range, THEN THE CrashSense SHALL raise a Pydantic validation error indicating the offending field and reason, and SHALL NOT create the instance.
5. WHEN the CrashEvent model is serialized to JSON, THE CrashSense SHALL emit every field listed in criterion 2 using the exact field names specified, with `timestamp` rendered as an ISO 8601 string in UTC.

### Requirement 13: Dashboard Layout

**User Story:** As an operator, I want a full-screen monitoring dashboard, so that I can observe sensors, crash events, and drone status at a glance during a live demo.

#### Acceptance Criteria

1. THE Dashboard SHALL render the Map_View occupying 80 percent (±1 percent) of the viewport width and 100 percent of the viewport height below the top status bar.
2. WHEN the Dashboard loads and no Alert is active, THE Dashboard SHALL render the Alert_Panel collapsed off-screen to the right of the viewport with 0 percent visible width.
3. WHEN a CrashEvent with status "DETECTED" is received, THE Dashboard SHALL slide the Alert_Panel in from the right edge to occupy 20 percent (±1 percent) of the viewport width within 300 milliseconds.
4. WHILE no CrashEvent with status "DETECTED" is active, THE Dashboard SHALL render the top status bar displaying the text "● MONITORING" in green with a text-to-background contrast ratio of at least 4.5:1.
5. WHEN a CrashEvent with status "DETECTED" is received, THE Dashboard SHALL update the top status bar to display the text "⚠ CRASH DETECTED" in red with a text-to-background contrast ratio of at least 4.5:1, within 500 milliseconds of receipt.
6. IF the Dashboard fails to receive or render an incoming CrashEvent within 2 seconds of its emission, THEN THE Dashboard SHALL retain the previous status bar state and display a non-blocking error indication that telemetry is delayed, without discarding queued events.
7. THE Dashboard SHALL be styled using Tailwind CSS layout utilities and SHALL render without horizontal scrolling at viewport widths from 1024 pixels to 1920 pixels.

### Requirement 14: Sensor Map Rendering

**User Story:** As an operator, I want all three sensors visible on the map at startup, so that I can confirm the monitoring network is correctly displayed before any event occurs.

#### Acceptance Criteria

1. THE Map_View SHALL be implemented in `frontend/src/components/Map.jsx` using Mapbox GL JS via the `mapbox-gl` npm package.
2. WHEN the Dashboard loads and `GET /sensors` returns successfully within 5 seconds, THE Map_View SHALL render all three sensors as pulsing blue dots with text labels showing each sensor's identifier within 2 seconds of receiving the response.
3. THE Map_View SHALL render Toll_Plaza sensors using a drone dock SVG icon sized 32x32 pixels in place of the pulsing blue dot, while retaining the text label.
4. IF the Mapbox API key is missing, invalid, or rejected by the Mapbox service, THEN THE Map_View SHALL fall back to a Leaflet.js map rendered over OpenStreetMap tiles within 3 seconds, preserve the same sensor rendering behavior defined in criteria 2 and 3, and display a non-blocking notice indicating that the fallback map provider is in use.
5. IF `GET /sensors` fails, returns a non-success response, returns fewer than three sensors, or does not respond within 5 seconds, THEN THE Map_View SHALL display an error indicator on the map identifying that sensor data could not be loaded and SHALL retry the request up to 3 times at 5-second intervals before remaining in the error state.

### Requirement 15: Crash Event Visualization

**User Story:** As an operator, I want each crash event drawn as a distinct map marker with a propagating wavefront, so that I can immediately locate the incident on the map.

#### Acceptance Criteria

1. WHEN a CrashEvent message is received over the WebSocket containing valid `crash_lat` (-90.0 to 90.0) and `crash_lon` (-180.0 to 180.0), THE Map_View SHALL drop a red pin at `(crash_lat, crash_lon)` within 500 milliseconds of message receipt and play a bounce animation lasting 1 second.
2. WHEN a CrashEvent message is received over the WebSocket with valid coordinates, THE Map_View SHALL render a dashed circle centered at `(crash_lat, crash_lon)` that expands from a radius of 0 meters to 500 meters over 3 seconds and then disappears.
3. WHEN a red pin is dropped for a CrashEvent, THE Map_View SHALL display a coordinate label adjacent to the pin showing `crash_lat` and `crash_lon` each formatted to 6 decimal places.
4. IF a CrashEvent message is received with `crash_lat` outside -90.0 to 90.0 or `crash_lon` outside -180.0 to 180.0 or with either field missing, THEN THE Map_View SHALL discard the message without rendering a pin, label, or wavefront and SHALL display an error indicator notifying the operator that an invalid crash event was received.
5. WHILE more than 50 active CrashEvent markers are present on the Map_View, THE Map_View SHALL remove the oldest markers first so that no more than 50 pins, labels, and wavefronts are rendered concurrently.

### Requirement 16: Drone Dispatch Animation

**User Story:** As a judge, I want to see a drone fly from the nearest toll plaza to the crash site in real time, so that the autonomous dispatch behavior is visible during the demo.

#### Acceptance Criteria

1. THE Drone_Tracker SHALL be implemented in `frontend/src/components/DroneTracker.jsx`.
2. WHEN a CrashEvent with status `"DETECTED"` is received containing a non-empty `nearest_sensor` value that resolves to a known Toll_Plaza with valid latitude in the range -90 to 90 and longitude in the range -180 to 180, THE Drone_Tracker SHALL select the drone origin as the Toll_Plaza identified by `nearest_sensor` and store its coordinates as `(drone_origin_lat, drone_origin_lon)` within 500 milliseconds of receipt.
3. IF a CrashEvent with status `"DETECTED"` is received with a missing, empty, or unresolvable `nearest_sensor`, THEN THE Drone_Tracker SHALL skip dispatch animation, display a visible indicator on the crash pin showing that no drone origin is available, and not emit a `"DRONE_ARRIVED"` update.
4. WHEN the drone origin has been selected per criterion 2, THE Drone_Tracker SHALL animate the drone icon from `(drone_origin_lat, drone_origin_lon)` to `(crash_lat, crash_lon)` over 8 seconds (8000 milliseconds, tolerance ±200 milliseconds) using `requestAnimationFrame` with linear interpolation of latitude and longitude.
5. WHILE the drone is in transit, THE Drone_Tracker SHALL display a live ETA countdown in whole seconds derived from the remaining animation time, refreshed at least once per second, with values monotonically non-increasing from 8 down to 0.
6. WHEN the drone icon reaches the crash pin (interpolation progress equals 1.0), THE Drone_Tracker SHALL render a green pulse animation at the crash location for a duration of 2 seconds with at least 2 visible pulse cycles.
7. WHEN the drone icon reaches the crash pin, THE Drone_Tracker SHALL send a CrashEvent update with status `"DRONE_ARRIVED"` over the WebSocket channel within 500 milliseconds of arrival.
8. IF the WebSocket channel is not open at the moment of arrival, THEN THE Drone_Tracker SHALL retain the `"DRONE_ARRIVED"` update in a pending state, retry transmission up to 3 times at 1 second intervals once the channel is open, and surface a visible error indicator on the crash pin if all retries fail.

### Requirement 17: Alert Panel

**User Story:** As an operator, I want a chronological list of crash events with status badges, so that I can review every incident and its dispatch progress.

#### Acceptance Criteria

1. THE Alert_Panel SHALL be implemented in `frontend/src/components/AlertPanel.jsx` as a vertically scrollable panel that retains up to 100 most recent CrashEvent cards in newest-first order and discards older cards beyond that limit.
2. WHEN a CrashEvent is received, THE Alert_Panel SHALL prepend a new card to the top of the panel within 500 milliseconds of receipt, displaying the event timestamp formatted as `YYYY-MM-DD HH:MM:SS` in 24-hour local time, the confidence value rendered as a percentage with one decimal place (e.g., `87.5%`), the GPS coordinate formatted as `lat, lon` with six decimal places each, the nearest sensor name as a non-empty string, and the current drone status badge.
3. WHILE a CrashEvent has status `"DETECTED"`, THE Alert_Panel SHALL render that card's status badge with a yellow background and the literal label text `DETECTING`.
4. WHILE a CrashEvent has status `"DRONE_DISPATCHED"`, THE Alert_Panel SHALL render that card's status badge with an orange background and the literal label text `DRONE DISPATCHED`.
5. WHILE a CrashEvent has status `"DRONE_ARRIVED"`, THE Alert_Panel SHALL render that card's status badge with a green background and the literal label text `DRONE ARRIVED`.
6. IF a CrashEvent has a status value other than `"DETECTED"`, `"DRONE_DISPATCHED"`, or `"DRONE_ARRIVED"`, THEN THE Alert_Panel SHALL render that card's status badge with a neutral gray background and the literal label text `UNKNOWN` while still displaying the remaining card fields.
7. WHILE no CrashEvent has been received since panel mount, THE Alert_Panel SHALL display an empty-state message indicating that no crash events have been detected.
8. WHEN the operator activates the `Simulate Crash` button, THE Alert_Panel SHALL issue a `POST /simulate-crash` request with a coordinate uniformly selected at random from inside the Sensor_Triangle and SHALL disable the button until the request completes or fails.
9. IF the `POST /simulate-crash` request returns a non-success response or fails to complete within 5 seconds, THEN THE Alert_Panel SHALL re-enable the `Simulate Crash` button and SHALL display a visible error indicator informing the operator that the simulation request failed.

### Requirement 18: Audio Monitor Visualization

**User Story:** As a judge, I want to see an audio waveform that flatlines and then spikes at the crash impact, so that the audio trigger is visually correlated with the detected event.

#### Acceptance Criteria

1. THE Audio_Monitor SHALL be implemented in `frontend/src/components/AudioMonitor.jsx` using the Web Audio API `AnalyserNode`.
2. WHEN the Dashboard loads, THE Audio_Monitor SHALL load a pre-recorded dashcam audio clip without requesting microphone access and SHALL begin rendering the waveform within 1 second of the clip becoming playable.
3. WHILE the dashcam clip plays through the pre-impact section, THE Audio_Monitor SHALL render a waveform whose absolute amplitude remains at or below 0.1 on a normalized scale of 0.0 to 1.0, refreshed at a minimum of 30 frames per second.
4. WHEN the dashcam clip reaches the impact moment, THE Audio_Monitor SHALL render a waveform spike whose peak absolute amplitude reaches at least 0.7 on the normalized scale of 0.0 to 1.0, with the visual peak occurring within 100 milliseconds of the corresponding audio sample.
5. IF the dashcam audio clip fails to load or decode, THEN THE Audio_Monitor SHALL display a static flat waveform and an inline error indication that the audio source is unavailable, without blocking the rest of the Dashboard from rendering.

### Requirement 19: WebSocket Client Hook

**User Story:** As a frontend developer, I want a reusable WebSocket hook with auto-reconnect, so that the Dashboard recovers gracefully from transient connection drops during a demo.

#### Acceptance Criteria

1. THE WebSocket_Client SHALL be implemented as a hook `useWebSocket` in `frontend/src/hooks/useWebSocket.js`.
2. WHEN a component using the WebSocket_Client mounts, THE WebSocket_Client SHALL open a WebSocket connection to `ws://localhost:8000/ws/events` within 5 seconds and SHALL set `isConnected` to true upon successful handshake.
3. WHEN the WebSocket_Client receives a message over the open connection, THE WebSocket_Client SHALL parse the message as JSON, append the parsed object to the `events` array, and update `latestEvent` to the parsed object, retaining at most the 100 most recent events and discarding the oldest entries when the limit is exceeded.
4. THE WebSocket_Client SHALL return an object containing `events` (array of parsed event objects in arrival order), `isConnected` (boolean reflecting current connection state), and `latestEvent` (the most recently received parsed event, or null if no events have been received).
5. IF the WebSocket connection drops or fails to establish, THEN THE WebSocket_Client SHALL set `isConnected` to false and attempt reconnection using exponential backoff with a multiplier of 2, starting at a 1 second delay and capped at a 30 second delay, continuing until the component unmounts or a connection is re-established.
6. IF an incoming message cannot be parsed as JSON, THEN THE WebSocket_Client SHALL discard that message, leave the `events` array and `latestEvent` unchanged, and SHALL NOT close the WebSocket connection.
7. WHEN the user clicks the `Replay Last Event` button on the Dashboard, THE Dashboard SHALL re-render the CrashEvent stored in `latestEvent` from local state without opening a new WebSocket connection and without requesting a new message from the backend.
8. IF the user clicks the `Replay Last Event` button while `latestEvent` is null, THEN THE Dashboard SHALL display an indication that no event is available to replay and SHALL NOT alter the rendered CrashEvent state.

### Requirement 20: End-to-End Demo Orchestration

**User Story:** As a presenter, I want a single command to drive the full pipeline, so that the demo runs reliably without manual coordination across components.

#### Acceptance Criteria

1. THE Demo_Runner SHALL be implemented in `backend/demo_runner.py`.
2. WHEN the Demo_Runner is executed, THE Demo_Runner SHALL load the trained Audio_Detector checkpoint from `backend/audio_model/crash_detector.pth`.
3. IF the Audio_Detector checkpoint file is not present at `backend/audio_model/crash_detector.pth` when the Demo_Runner is executed, THEN THE Demo_Runner SHALL terminate with a non-zero exit status and emit an error message indicating the missing checkpoint path.
4. WHEN the Demo_Runner plays the dashcam crash audio through the Audio_Detector and the Audio_Detector returns `event = "CRASH"`, THE Demo_Runner SHALL select a uniformly random coordinate strictly inside the Sensor_Triangle as the simulated crash location.
5. WHEN a simulated crash location is selected, THE Demo_Runner SHALL invoke the Triangulation_Solver to produce noisy arrival times and recover an estimated coordinate.
6. WHEN a CrashEvent has been constructed, THE Demo_Runner SHALL POST the CrashEvent to the Backend_API using `httpx`.
7. WHEN the Backend_API receives a POSTed CrashEvent from the Demo_Runner, THE Backend_API SHALL broadcast the CrashEvent to all connected Dashboard WebSocket clients.
8. IF the POST from the Demo_Runner to the Backend_API fails to connect or returns a non-success response, THEN THE Demo_Runner SHALL retry the POST up to 2 additional times with a 500 millisecond delay between attempts and, upon final failure, terminate with a non-zero exit status and an error message indicating the POST failure.
9. WHEN the Demo_Runner is invoked, THE CrashSense SHALL complete the pipeline from the start of audio playback to Dashboard CrashEvent rendering within 3.0 seconds measured end-to-end.
10. THE CrashSense SHALL provide a `start.sh` or `Makefile` target that launches three processes consisting of the uvicorn Backend_API on port 8000, the frontend dev server via `npm run dev`, and the Demo_Runner.

### Requirement 21: Backend Dependencies

**User Story:** As a developer setting up the project, I want a single dependency manifest for the backend, so that I can install the exact set of libraries the system requires.

#### Acceptance Criteria

1. THE CrashSense SHALL provide a `backend/requirements.txt` file listing the following libraries: `torch`, `torchvision`, `torchaudio`, `librosa`, `matplotlib`, `numpy`, `scipy`, `fastapi`, `uvicorn`, `websockets`, `httpx`, `geopy`, `pydantic`, `python-multipart`, and `audioset-download`.
2. THE CrashSense SHALL pin each library in `backend/requirements.txt` to an exact version using the `==` specifier so that repeated installations on the same Python interpreter resolve to identical package versions.
3. WHEN a developer runs `pip install -r backend/requirements.txt` in a clean Python 3.10 or newer virtual environment with network access, THE CrashSense SHALL allow installation to complete with a non-error exit status and with every listed library importable in a Python REPL.
4. IF `pip install -r backend/requirements.txt` terminates with a non-zero exit status, THEN THE CrashSense SHALL surface the failing package name and the underlying pip error output to the developer through standard error without modifying the existing virtual environment's installed packages.
5. IF `backend/requirements.txt` is missing any of the libraries listed in criterion 1, contains a duplicate library entry, or omits the `==` version pin for any listed library, THEN THE CrashSense SHALL be considered non-compliant with this requirement.

### Requirement 22: Frontend Dependencies

**User Story:** As a developer setting up the project, I want a single dependency manifest for the frontend, so that I can install the exact set of libraries the Dashboard requires.

#### Acceptance Criteria

1. THE CrashSense SHALL provide a `frontend/package.json` file that declares, under the `dependencies` field, `mapbox-gl` at version `^3.0.0`, `@mapbox/mapbox-gl-geocoder` at version `^5.0.0`, `react` at version `^18.0.0`, `react-dom` at version `^18.0.0`, and `tailwindcss` at version `^3.0.0`.
2. THE CrashSense SHALL declare in `frontend/package.json` an `engines.node` field requiring Node.js version `>=18.0.0` and an `install` script that completes successfully within 300 seconds on a clean checkout with network access.
3. WHEN a developer runs `npm install` in the `frontend/` directory on a clean checkout, THE CrashSense SHALL install all declared dependencies and produce a `node_modules/` directory containing each of `mapbox-gl`, `@mapbox/mapbox-gl-geocoder`, `react`, `react-dom`, and `tailwindcss` with installed versions matching their declared semver ranges.
4. IF `npm install` fails to resolve or download any declared dependency, THEN THE CrashSense SHALL cause the `npm install` command to exit with a non-zero status code and emit an error message identifying the unresolved package name and the requested version range, while leaving any previously installed `node_modules/` contents unchanged.
5. IF `frontend/package.json` is missing any of the five required dependency entries or specifies a version range outside the declared major versions (mapbox-gl 3.x, @mapbox/mapbox-gl-geocoder 5.x, react 18.x, react-dom 18.x, tailwindcss 3.x), THEN THE CrashSense SHALL be considered non-conformant to this requirement and the manifest validation step SHALL exit with a non-zero status code indicating which dependency entry is missing or out of range.

### Requirement 23: Judging-Moment Demo Outcomes

**User Story:** As a judge, I want six specific demo outcomes to be observable in sequence, so that I can verify the complete pipeline functions during a live demonstration.

#### Acceptance Criteria

1. WHEN the demo crash audio file is played as input to the Audio_Detector, THE Audio_Detector SHALL print to standard output, within 2.0 seconds of audio playback completion, a single terminal line of the form `CRASH DETECTED — <value>% confidence` where `<value>` is a numeric value between 85.0 and 100.0 inclusive, formatted to one decimal place.
2. IF the Audio_Detector computes a crash confidence value below 85.0 for the demo crash audio, THEN THE Audio_Detector SHALL print an error message indicating that the demo confidence threshold was not met and SHALL NOT print the `CRASH DETECTED` line.
3. WHEN `tdoa_demo.py` is executed, THE Triangulation_Solver SHALL print to standard output, within 5.0 seconds of execution start, the true coordinate as a latitude/longitude pair, the solved coordinate as a latitude/longitude pair, and a localization error value in meters that is greater than or equal to 0.0 and less than or equal to 30.0.
4. WHEN the Dashboard finishes loading in the browser, THE Map_View SHALL render exactly three sensor dots, each fully contained within the visible map viewport bounds, within 3.0 seconds of the Dashboard load-complete event.
5. WHEN the operator clicks the `Simulate Crash` button, THE Map_View SHALL drop a red pin with a visible coordinate label showing latitude and longitude on the map within 1.0 second of the click event being registered.
6. WHEN a CrashEvent is rendered on the Map_View, THE Drone_Tracker SHALL animate the drone icon along a continuous path from the nearest Toll_Plaza to the red pin and SHALL position the drone icon at the red pin coordinates at an elapsed animation time between 7.5 seconds and 8.5 seconds inclusive, measured from the start of the animation.
7. WHEN a CrashEvent progresses through dispatch, THE Alert_Panel SHALL display a single card whose status badge transitions in order through the states `DETECTING`, then `DRONE DISPATCHED`, then `DRONE ARRIVED`, with each subsequent state replacing the previous state and no state being skipped or repeated.
8. IF the Drone_Tracker fails to position the drone icon at the red pin within 8.5 seconds of animation start, THEN THE Alert_Panel SHALL display an error indication on the CrashEvent card and SHALL NOT transition the status badge to `DRONE ARRIVED`.
