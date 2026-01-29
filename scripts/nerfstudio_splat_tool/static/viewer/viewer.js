import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { FirstPersonControls } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/FirstPersonControls.js";
import { PLYLoader } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/PLYLoader.js";

const canvas = document.getElementById("viewport");
const statusEl = document.getElementById("status");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0.5, 2);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);

const ambient = new THREE.AmbientLight(0xffffff, 0.8);
scene.add(ambient);

const directional = new THREE.DirectionalLight(0xffffff, 0.6);
directional.position.set(2, 4, 3);
scene.add(directional);

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;

const fps = new FirstPersonControls(camera, renderer.domElement);
fps.movementSpeed = 2.5;
fps.lookSpeed = 0.08;
fps.enabled = true;

let activeControls = fps;

window.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() === "o") {
    activeControls = activeControls === fps ? orbit : fps;
    statusEl.textContent = `Controls: ${activeControls === fps ? "First-person" : "Orbit"}`;
  }
});

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener("resize", onResize);

function parseSceneUrl() {
  const params = new URLSearchParams(window.location.search);
  return params.get("scene");
}

function fitCamera(object) {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3()).length();
  const center = box.getCenter(new THREE.Vector3());
  const distance = size * 0.75 || 2;
  camera.position.copy(center.clone().add(new THREE.Vector3(0, 0.5, distance)));
  camera.lookAt(center);
  orbit.target.copy(center);
}

async function loadScene(url) {
  if (!url) {
    statusEl.textContent = "No scene specified.";
    return;
  }

  if (url.toLowerCase().endsWith(".splat")) {
    statusEl.textContent = "Gaussian splat file detected. Viewer currently expects .ply output.";
  }

  const loader = new PLYLoader();
  loader.load(
    url,
    (geometry) => {
      geometry.computeVertexNormals();
      const material = new THREE.PointsMaterial({ size: 0.01, color: 0xffffff });
      const points = new THREE.Points(geometry, material);
      scene.add(points);
      fitCamera(points);
      statusEl.textContent = "Scene loaded.";
    },
    undefined,
    (error) => {
      console.error(error);
      statusEl.textContent = "Failed to load scene.";
    },
  );
}

function animate() {
  requestAnimationFrame(animate);
  if (activeControls === fps) {
    fps.update(0.015);
  } else {
    orbit.update();
  }
  renderer.render(scene, camera);
}

loadScene(parseSceneUrl());
animate();
