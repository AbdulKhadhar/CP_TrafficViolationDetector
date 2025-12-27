from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.db.models.functions import ExtractHour, TruncDate
from datetime import datetime, timedelta
import cv2
import json
import threading

from .models import Camera, Violation, DetectionLog, SystemSettings
from .ai_models.violation_processor import ViolationProcessor

# Global variable to track active detection threads
active_detections = {}

def home(request):
    """Dashboard homepage"""
    # Get statistics
    total_violations = Violation.objects.count()
    pending_violations = Violation.objects.filter(status='PENDING').count()
    confirmed_violations = Violation.objects.filter(status='CONFIRMED').count()
    active_cameras = Camera.objects.filter(is_active=True).count()
    
    # Recent violations
    recent_violations = Violation.objects.all()[:10]
    
    # Violations by type
    violations_by_type = Violation.objects.values('violation_type').annotate(
        count=Count('id')
    )
    
    # Violations today
    today = datetime.now().date()
    violations_today = Violation.objects.filter(
        detected_at__date=today
    ).count()
    
    context = {
        'total_violations': total_violations,
        'pending_violations': pending_violations,
        'confirmed_violations': confirmed_violations,
        'active_cameras': active_cameras,
        'recent_violations': recent_violations,
        'violations_by_type': violations_by_type,
        'violations_today': violations_today,
    }
    
    return render(request, 'detector/home.html', context)

def violations_list(request):
    """List all violations with filters"""
    violations = Violation.objects.all()
    
    # Filters
    violation_type = request.GET.get('type')
    status = request.GET.get('status')
    camera_id = request.GET.get('camera')
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')
    
    if violation_type:
        violations = violations.filter(violation_type=violation_type)
    
    if status:
        violations = violations.filter(status=status)
    
    if camera_id:
        violations = violations.filter(camera_id=camera_id)
    
    if date_from:
        violations = violations.filter(detected_at__date__gte=date_from)
    
    if date_to:
        violations = violations.filter(detected_at__date__lte=date_to)
    
    # Pagination
    paginator = Paginator(violations, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    cameras = Camera.objects.all()
    
    context = {
        'page_obj': page_obj,
        'cameras': cameras,
        'violation_type': violation_type,
        'status': status,
        'camera_id': camera_id,
    }
    
    return render(request, 'detector/violations.html', context)

def violation_detail(request, pk):
    """View and update individual violation"""
    violation = get_object_or_404(Violation, pk=pk)
    
    if request.method == 'POST':
        new_status = request.POST.get('status')
        notes = request.POST.get('notes')
        
        if new_status:
            violation.status = new_status
            violation.notes = notes
            violation.reviewed_at = datetime.now()
            violation.reviewed_by = request.user.username if request.user.is_authenticated else 'Admin'
            violation.save()
            
            messages.success(request, 'Violation updated successfully')
            return redirect('violation_detail', pk=pk)
    
    context = {
        'violation': violation,
    }
    
    return render(request, 'detector/violation_detail.html', context)

@csrf_exempt
def upload_video(request):
    """Upload and process video file"""
    if request.method == 'POST':
        video_file = request.FILES.get('video')
        camera_id = request.POST.get('camera_id')
        
        if not video_file or not camera_id:
            return JsonResponse({'error': 'Missing video or camera'}, status=400)
        
        # Save video file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            for chunk in video_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        
        # Process video in background thread
        def process_video_task():
            try:
                processor = ViolationProcessor(camera_id=camera_id)
                violations = processor.process_video_file(tmp_path)
                print(f"Processed video: {violations} violations detected")
            except Exception as e:
                print(f"Error processing video: {e}")
            finally:
                import os
                os.unlink(tmp_path)
        
        thread = threading.Thread(target=process_video_task)
        thread.start()
        
        return JsonResponse({
            'success': True,
            'message': 'Video uploaded and processing started'
        })
    
    cameras = Camera.objects.filter(is_active=True)
    return render(request, 'detector/upload_video.html', {'cameras': cameras})

def start_live_detection(request, camera_id):
    """Start live detection for a camera"""
    camera = get_object_or_404(Camera, id=camera_id)
    
    if camera_id in active_detections:
        messages.warning(request, 'Detection already running for this camera')
        return redirect('live_feed', camera_id=camera_id)
    
    # Start detection in background thread
    def detection_task():
        try:
            processor = ViolationProcessor(camera_id=camera_id)
            processor.process_live_stream(duration_seconds=3600)  # 1 hour
        except Exception as e:
            print(f"Detection error: {e}")
        finally:
            if camera_id in active_detections:
                del active_detections[camera_id]
    
    thread = threading.Thread(target=detection_task)
    thread.daemon = True
    thread.start()
    
    active_detections[camera_id] = thread
    
    messages.success(request, f'Live detection started for {camera.name}')
    return redirect('live_feed', camera_id=camera_id)

def stop_live_detection(request, camera_id):
    """Stop live detection for a camera"""
    if camera_id in active_detections:
        # Thread will stop on its own, just remove from dict
        del active_detections[camera_id]
        messages.success(request, 'Detection stopped')
    
    return redirect('home')

def live_feed(request, camera_id):
    """Display live feed from camera"""
    camera = get_object_or_404(Camera, id=camera_id)
    
    # Get recent violations for this camera
    recent_violations = Violation.objects.filter(
        camera=camera
    ).order_by('-detected_at')[:10]
    
    is_active = camera_id in active_detections
    
    context = {
        'camera': camera,
        'recent_violations': recent_violations,
        'is_active': is_active,
    }
    
    return render(request, 'detector/live_feed.html', context)

def analytics(request):
    """Analytics and reporting"""
    # Get date range
    days = int(request.GET.get('days', 7))
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Violations over time - SQLite compatible
    violations_by_day = Violation.objects.filter(
        detected_at__gte=start_date
    ).annotate(
        day=TruncDate('detected_at')
    ).values('day').annotate(count=Count('id')).order_by('day')
    
    # Violations by type
    violations_by_type = Violation.objects.filter(
        detected_at__gte=start_date
    ).values('violation_type').annotate(count=Count('id'))
    
    # Violations by camera
    violations_by_camera = Violation.objects.filter(
        detected_at__gte=start_date
    ).values('camera__name').annotate(count=Count('id'))
    
    # Top violation hours - SQLite compatible
    from django.db.models.functions import ExtractHour
    violations_by_hour = Violation.objects.filter(
        detected_at__gte=start_date
    ).annotate(
        hour=ExtractHour('detected_at')
    ).values('hour').annotate(count=Count('id')).order_by('hour')
    
    context = {
        'days': days,
        'violations_by_day': list(violations_by_day),
        'violations_by_type': list(violations_by_type),
        'violations_by_camera': list(violations_by_camera),
        'violations_by_hour': list(violations_by_hour),
    }
    
    return render(request, 'detector/analytics.html', context)

def api_violations(request):
    """API endpoint for violations data"""
    camera_id = request.GET.get('camera')
    limit = int(request.GET.get('limit', 100))
    
    violations = Violation.objects.all()
    
    if camera_id:
        violations = violations.filter(camera_id=camera_id)
    
    violations = violations[:limit]
    
    data = [{
        'id': v.id,
        'type': v.violation_type,
        'camera': v.camera.name,
        'time': v.detected_at.isoformat(),
        'confidence': v.confidence_score,
        'status': v.status,
        'plate': v.license_plate,
    } for v in violations]
    
    return JsonResponse(data, safe=False)

def cameras_list(request):
    """List all cameras with monitoring controls"""
    cameras = Camera.objects.all()
    
    # Add active detection status to each camera
    for camera in cameras:
        camera.detection_active = camera.id in active_detections
        
        # Get recent violations count
        camera.violations_today = Violation.objects.filter(
            camera=camera,
            detected_at__date=datetime.now().date()
        ).count()
        
        # Get last detection time
        last_violation = Violation.objects.filter(camera=camera).first()
        camera.last_detection = last_violation.detected_at if last_violation else None
    
    context = {
        'cameras': cameras,
    }
    
    return render(request, 'detector/cameras_list.html', context)