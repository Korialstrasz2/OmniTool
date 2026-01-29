const statusEl = document.getElementById("status");
const canvas = document.getElementById("viewer");

const params = new URLSearchParams(window.location.search);
const sceneUrl = params.get("scene");

function setStatus(message) {
  statusEl.textContent = message;
}

async function loadSplat(url) {
  setStatus("Loading Gaussian splat scene...");
  const moduleUrl =
    "https://unpkg.com/@mkkellogg/gaussian-splats-3d@0.6.3/build/gaussian-splats-3d.module.js";

  const module = await import(moduleUrl);
  const Viewer = module.Viewer || module.GaussianSplattingViewer || module.GaussianSplats3DViewer;
  if (!Viewer) {
    throw new Error("Gaussian splat viewer module did not expose a Viewer class.");
  }

  const viewer = new Viewer({
    canvas,
    useAdaptiveDownsampling: true,
    cameraUp: [0, -1, 0],
    initialCameraPosition: [0, -1, -2],
    initialCameraLookAt: [0, 0, 0],
  });

  await viewer.addSplatScene(url, {
    progressiveLoad: true,
    showLoadingUI: false,
    splatAlphaRemovalThreshold: 0.3,
  });

  viewer.start();
  setStatus("Splat scene ready. Use WASD + mouse to move.");
}

async function loadPly(url) {
  setStatus("Loading PLY point cloud...");
  const THREE = await import("https://unpkg.com/three@0.160.0/build/three.module.js");
  const { OrbitControls } = await import(
    "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js"
  );
  const { PLYLoader } = await import(
    "https://unpkg.com/three@0.160.0/examples/jsm/loaders/PLYLoader.js"
  );

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0f1a);

  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 1000);
  camera.position.set(0, 0.5, 2);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const loader = new PLYLoader();
  loader.load(url, (geometry) => {
    geometry.computeVertexNormals();
    const material = new THREE.PointsMaterial({ size: 0.01, vertexColors: true });
    const points = new THREE.Points(geometry, material);
    scene.add(points);
    setStatus("PLY scene ready. Use orbit controls to navigate.");
  });

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }

  window.addEventListener("resize", () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  });

  animate();
}

async function init() {
  if (!sceneUrl) {
    setStatus("Waiting for a scene file...");
    return;
  }

  const lower = sceneUrl.toLowerCase();
  try {
    if (lower.endsWith(".splat")) {
      await loadSplat(sceneUrl);
    } else if (lower.endsWith(".ply")) {
      await loadPly(sceneUrl);
    } else {
      setStatus("Unsupported scene format. Use .splat or .ply files.");
    }
  } catch (error) {
    console.error(error);
    setStatus("Failed to load scene. Check browser console for details.");
  }
}

init();
