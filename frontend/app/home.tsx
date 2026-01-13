import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'expo-router';

export default function HomeScreen() {
  const { user, logout } = useAuth();
  const router = useRouter();

  const handleLogout = () => {
    Alert.alert(
      'Logout',
      'Are you sure you want to logout?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Logout',
          style: 'destructive',
          onPress: async () => {
            await logout();
            router.replace('/login' as any);
          },
        },
      ]
    );
  };

  const handlePrediction = () => {
    router.push('/prediction' as any);
  };

  return (
    <LinearGradient
      colors={['#56ab2f', '#a8e063']}
      style={styles.container}
    >
      <View style={styles.content}>
        <View style={styles.header}>
          <Text style={styles.emoji}>🌿</Text>
          <Text style={styles.welcomeText}>Welcome,</Text>
          <Text style={styles.userName}>{user?.name || 'User'}</Text>
          <Text style={styles.subtitle}>What would you like to do today?</Text>
        </View>

        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={styles.card}
            onPress={handlePrediction}
            activeOpacity={0.8}
          >
            <View style={styles.cardContent}>
              <Text style={styles.cardEmoji}>🔬</Text>
              <Text style={styles.cardTitle}>Plant Disease Detection</Text>
              <Text style={styles.cardDescription}>
                Upload a plant leaf image to detect diseases
              </Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.card, styles.logoutCard]}
            onPress={handleLogout}
            activeOpacity={0.8}
          >
            <View style={styles.cardContent}>
              <Text style={styles.cardEmoji}>👋</Text>
              <Text style={styles.cardTitle}>Logout</Text>
              <Text style={styles.cardDescription}>
                Sign out from your account
              </Text>
            </View>
          </TouchableOpacity>
        </View>

        <View style={styles.footer}>
          <Text style={styles.footerText}>Smart Leaf v1.0</Text>
        </View>
      </View>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
    padding: 20,
    paddingTop: 60,
  },
  header: {
    alignItems: 'center',
    marginBottom: 40,
  },
  emoji: {
    fontSize: 64,
    marginBottom: 16,
  },
  welcomeText: {
    fontSize: 24,
    color: '#fff',
    fontWeight: '300',
  },
  userName: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 12,
  },
  subtitle: {
    fontSize: 16,
    color: '#f0f0f0',
    textAlign: 'center',
  },
  buttonContainer: {
    flex: 1,
    justifyContent: 'center',
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 20,
    padding: 24,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 6,
  },
  logoutCard: {
    backgroundColor: '#f8f8f8',
  },
  cardContent: {
    alignItems: 'center',
  },
  cardEmoji: {
    fontSize: 48,
    marginBottom: 12,
  },
  cardTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  cardDescription: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
  footer: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  footerText: {
    color: '#fff',
    fontSize: 12,
    opacity: 0.8,
  },
});
