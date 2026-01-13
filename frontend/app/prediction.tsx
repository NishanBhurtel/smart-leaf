import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  Platform,
  ScrollView,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { Image } from "expo-image";
import { LinearGradient } from 'expo-linear-gradient';
import { useRouter } from 'expo-router';
import { API_ENDPOINTS } from '../config/api';

interface PredictionResponse {
  predicted_class: string;
  confidence: number;
  all_confidences: {
    [key: string]: number;
  };
}

export default function PredictionScreen() {
  const [image, setImage] = useState<{ uri: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const router = useRouter();

  useEffect(() => {
    (async () => {
      const galleryStatus = await ImagePicker.requestMediaLibraryPermissionsAsync();
      const cameraStatus = await ImagePicker.requestCameraPermissionsAsync();
      
      setHasPermission(galleryStatus.status === "granted" && cameraStatus.status === "granted");
      
      if (galleryStatus.status !== "granted" || cameraStatus.status !== "granted") {
        Alert.alert(
          "Permissions required",
          "This app needs permission to access your camera and photo library."
        );
      }
    })();
  }, []);

  const pickImage = async () => {
    try {
      setResult(null);
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        quality: 0.8,
      });

      if (!result.canceled) {
        setImage({ uri: result.assets[0].uri });
      }
    } catch (err) {
      console.error("Image pick error:", err);
      Alert.alert("Error", "Could not open image picker.");
    }
  };

  const takePhoto = async () => {
    try {
      setResult(null);
      const result = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        quality: 0.8,
      });

      if (!result.canceled) {
        setImage({ uri: result.assets[0].uri });
      }
    } catch (err) {
      console.error("Camera error:", err);
      Alert.alert("Error", "Could not open camera.");
    }
  };

  const uploadImage = async () => {
    if (!image?.uri) {
      Alert.alert("No image", "Please pick an image first.");
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      const uri = image.uri;
      const filename = uri.split("/").pop() || "photo.jpg";
      // Try to infer MIME type
      const match = /\.(\w+)$/.exec(filename);
      const ext = match ? match[1].toLowerCase() : "jpg";
      const type = ext === "png" ? "image/png" : "image/jpeg";

      formData.append("file", {
        uri: Platform.OS === "ios" ? uri.replace("file://", "") : uri,
        name: filename,
        type,
      } as any);

      const response = await fetch(API_ENDPOINTS.PREDICT, {
        method: "POST",
        body: formData,
        headers: {
          Accept: "application/json",
        },
      });

      if (!response.ok) {
        let text;
        try {
          text = await response.text();
          const json = JSON.parse(text);
          throw new Error(json.error || JSON.stringify(json) || text);
        } catch (parseErr) {
          throw new Error(text || `Server returned ${response.status}`);
        }
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Upload error:", err);
      Alert.alert(
        "Upload failed", 
        (err instanceof Error ? err.message : String(err)) || "Unknown error occurred while uploading."
      );
    } finally {
      setLoading(false);
    }
  };

  const renderResult = () => {
    if (!result) return null;
    const { predicted_class, confidence, all_confidences } = result;
    return (
      <View style={styles.resultBox}>
        <Text style={styles.resultTitle}>🎯 Prediction Result</Text>
        <View style={styles.predictionCard}>
          <Text style={styles.predClass}>{predicted_class.replace(/_/g, " ")}</Text>
          <View style={styles.confidenceContainer}>
            <Text style={styles.confidenceLabel}>Confidence</Text>
            <Text style={styles.confidence}>
              {confidence?.toFixed?.(4) ?? confidence}%
            </Text>
          </View>
        </View>
        
        <Text style={styles.subTitle}>Top 5 Predictions:</Text>
        {all_confidences && (
          <View style={styles.confList}>
            {Object.entries(all_confidences)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 5)
              .map(([name, val], index) => (
                <View key={name} style={styles.confItemContainer}>
                  <View style={styles.confRank}>
                    <Text style={styles.confRankText}>{index + 1}</Text>
                  </View>
                  <View style={styles.confDetails}>
                    <Text style={styles.confName}>{name.replace(/_/g, " ")}</Text>
                    <Text style={styles.confValue}>{Number(val).toFixed(4)}%</Text>
                  </View>
                </View>
              ))}
          </View>
        )}
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#56ab2f', '#a8e063']}
        style={styles.headerGradient}
      >
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Text style={styles.backButtonText}>← Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Plant Disease Detection</Text>
        </View>
      </LinearGradient>

      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.buttonRow}>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={takePhoto}
            activeOpacity={0.8}
          >
            <LinearGradient
              colors={['#FF6B6B', '#EE5A6F']}
              style={styles.buttonGradient}
            >
              <Text style={styles.buttonIcon}>📸</Text>
              <Text style={styles.buttonText}>Take Photo</Text>
            </LinearGradient>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.actionButton}
            onPress={pickImage}
            activeOpacity={0.8}
          >
            <LinearGradient
              colors={['#4c669f', '#3b5998']}
              style={styles.buttonGradient}
            >
              <Text style={styles.buttonIcon}>�️</Text>
              <Text style={styles.buttonText}>Pick Image</Text>
            </LinearGradient>
          </TouchableOpacity>
        </View>

        <TouchableOpacity
          style={[styles.analyzeButton, (!image || loading) && styles.buttonDisabled]}
          onPress={uploadImage}
          disabled={!image || loading}
          activeOpacity={0.8}
        >
          <LinearGradient
            colors={['#56ab2f', '#a8e063']}
            style={styles.analyzeButtonGradient}
          >
            <Text style={styles.analyzeButtonIcon}>🔬</Text>
            <Text style={styles.analyzeButtonText}>Analyze Plant Disease</Text>
          </LinearGradient>
        </TouchableOpacity>

        {loading && (
          <View style={styles.loadingBox}>
            <ActivityIndicator size="large" color="#56ab2f" />
            <Text style={styles.loadingText}>Analyzing image...</Text>
          </View>
        )}

        {image && (
          <View style={styles.previewBox}>
            <Text style={styles.sectionTitle}>Selected Image</Text>
            <View style={styles.imageContainer}>
              <Image 
                source={{ uri: image.uri }} 
                style={styles.image} 
                contentFit="cover"
                transition={200}
              />
            </View>
          </View>
        )}

        {renderResult()}

        
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  headerGradient: {
    paddingTop: Platform.OS === 'ios' ? 50 : 20,
    paddingBottom: 20,
  },
  header: {
    paddingHorizontal: 20,
  },
  backButton: {
    marginBottom: 10,
  },
  backButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  actionButton: {
    flex: 1,
    marginHorizontal: 5,
    borderRadius: 15,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 4,
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonGradient: {
    paddingVertical: 16,
    alignItems: 'center',
  },
  buttonIcon: {
    fontSize: 24,
    marginBottom: 4,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  analyzeButton: {
    marginBottom: 20,
    borderRadius: 15,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.3,
    shadowRadius: 6,
    elevation: 6,
  },
  analyzeButtonGradient: {
    paddingVertical: 18,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
  },
  analyzeButtonIcon: {
    fontSize: 24,
    marginRight: 8,
  },
  analyzeButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  previewBox: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  imageContainer: {
    backgroundColor: '#fff',
    borderRadius: 15,
    padding: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  image: {
    width: '100%',
    height: 300,
    borderRadius: 12,
  },
  loadingBox: {
    backgroundColor: '#fff',
    borderRadius: 15,
    padding: 30,
    alignItems: 'center',
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#666',
  },
  resultBox: {
    backgroundColor: '#fff',
    borderRadius: 15,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  resultTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 16,
  },
  predictionCard: {
    backgroundColor: '#f8f8f8',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    borderLeftWidth: 4,
    borderLeftColor: '#56ab2f',
  },
  predClass: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
    textTransform: 'capitalize',
  },
  confidenceContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  confidenceLabel: {
    fontSize: 14,
    color: '#666',
  },
  confidence: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#56ab2f',
  },
  subTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  confList: {
    marginTop: 8,
  },
  confItemContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f8f8f8',
    borderRadius: 10,
    padding: 12,
    marginBottom: 8,
  },
  confRank: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: '#56ab2f',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  confRankText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14,
  },
  confDetails: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  confName: {
    flex: 1,
    fontSize: 14,
    color: '#333',
    textTransform: 'capitalize',
  },
  confValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#666',
  },
  footer: {
    marginTop: 20,
    marginBottom: 20,
    padding: 16,
    backgroundColor: '#fff',
    borderRadius: 10,
  },
  note: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
});
