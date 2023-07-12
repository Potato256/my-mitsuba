#include <mitsuba/render/scene.h>

MTS_NAMESPACE_BEGIN

class myPathIntegrator : public SamplingIntegrator {
public:
    MTS_DECLARE_CLASS()

private:
    int m_maxDepth;
    int m_rrDepth;
    static int m_LiCount;
    
public:

    /// Initialize the integrator with the specified properties
    myPathIntegrator(const Properties &props) : SamplingIntegrator(props) {
        m_maxDepth = props.getInteger("maxDepth", 50);
        m_rrDepth = props.getInteger("rrDepth", 0);

    }

    // Unserialize from a binary data stream
    myPathIntegrator(Stream *stream, InstanceManager *manager)
        : SamplingIntegrator(stream, manager) {}
    
    /// Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const {
        SamplingIntegrator::serialize(stream, manager);
    }
    
    /// Preprocess function -- called on the initiating machine
    bool preprocess(const Scene *scene, RenderQueue *queue,
        const RenderJob *job, int sceneResID, int cameraResID, 
        int samplerResID) {
        SamplingIntegrator::preprocess(scene, queue, job, sceneResID,
            cameraResID, samplerResID);
        return true;
    }

    /// Query for an unbiased estimate of the radiance along <tt>r</tt>
    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        ++m_LiCount;
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        if (its.isValid() && its.isEmitter())
            return its.Le(-ray.d);

        while(rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            if (!its.isValid())
                break;

            Float q = 0.8f;
            const BSDF *bsdf = its.getBSDF(ray);

            /* Estimate the direct illumination if this is requested */
            DirectSamplingRecord dRec(its);

            if (rRec.nextSample1D() > q) {
                throughput /= 1-q;
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                /* Allocate a record for querying the BSDF */
                BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                /* Evaluate BSDF * cos(theta) */
                const Spectrum bsdfVal = bsdf->eval(bRec);
                Li += throughput * value * bsdfVal;
                return Li;
            }
            throughput /= q;

            /* Sample BSDF * cos(theta) */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());

            const Vector wo = its.toWorld(bRec.wo);
            
            /* Keep track of the throughput and relative
               refractive index along the path */
            throughput *= bsdfWeight ;

            /* Trace a ray in this direction */
            ray = Ray(its.p, wo, ray.time);

            Spectrum value;
            if (scene->rayIntersect(ray, its)) {
                if (its.isEmitter()) {
                    return Li;
                }
            }
            ++rRec.depth;
        }
        return Li;
    }
};

int myPathIntegrator::m_LiCount = 0;

MTS_IMPLEMENT_CLASS_S(myPathIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(myPathIntegrator, "My path integrator");
MTS_NAMESPACE_END